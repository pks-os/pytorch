from __future__ import annotations

import traceback
import dataclasses
import hashlib
import logging
import os
import os.path
from typing import Dict, List, Optional, Tuple
from typing_extensions import override

import torch
from torch.utils._triton import has_triton_package

from ..remote_cache import (
    JsonDataTy,
    RemoteCache,
    RemoteCacheBackend,
    RemoteCacheJsonSerde,
)


if has_triton_package():
    from triton import Config

log = logging.getLogger(__name__)


_InductorMetaTy = Dict[str, object]


@dataclasses.dataclass
class AutotuneCache:
    configs_hash: str
    filename: str
    local_cache: Optional[Tuple[RemoteCache[JsonDataTy], str]] = None
    remote_cache: Optional[Tuple[RemoteCache[JsonDataTy], str]] = None

    # Create a AutotuneCache. Returns None if none of the caches can be used.
    @staticmethod
    def create(
        inductor_meta: _InductorMetaTy,
        # TODO: I think the filename is really a hash of the source graph - so it's
        # equal to saying "hash of source graph"?
        filename: str,
        configs_hash: str,
    ) -> Optional[AutotuneCache]:
        cache = AutotuneCache(configs_hash, filename)
        cache._setup_local_cache(inductor_meta, filename)
        cache._setup_remote_autotune_cache(inductor_meta, filename)
        if cache.local_cache or cache.remote_cache:
            return cache
        else:
            return None

    # Read the best config options from the most local cache and return it.
    def _read(self) -> Optional[Dict[str, JsonDataTy]]:
        if local_cache := self.local_cache:
            cache, key = local_cache
            if best_config := cache.get(key):
                if isinstance(best_config, dict):
                    return best_config

        if remote_cache := self.remote_cache:
            cache, key = remote_cache
            if best_config := cache.get(key):
                if isinstance(best_config, dict):
                    return best_config

        return None

    # Read the best config options from the most local cache and figure out
    # which `configs` represents that option.
    def read_best(
        self, inductor_meta: _InductorMetaTy, configs: List[Config]
    ) -> Optional[Config]:
        if best := self._read():
            return _load_cached_autotuning(
                best, self.configs_hash, configs, inductor_meta
            )
        return None

    # Set up local filesystem caching information
    def _setup_local_cache(self, inductor_meta: _InductorMetaTy, filename: str) -> None:
        if not inductor_meta.get("autotune_local_cache", True):
            return

        cache_filename = os.path.splitext(filename)[0] + ".best_config"
        local_cache = LocalAutotuneCache()
        self.local_cache = (local_cache, cache_filename)

    # Set up remote caching information
    def _setup_remote_autotune_cache(
        self, inductor_meta: _InductorMetaTy, filename: str
    ) -> None:
        if not _should_use_remote_autotune_cache(inductor_meta):
            return

        if (backend_hash := inductor_meta.get("backend_hash", None)) is None:
            log.debug(
                "backend_hash is not passed on the inductor_meta, unable to use autotune remote cache"
            )
            return
        assert isinstance(backend_hash, str)

        is_fbcode = bool(inductor_meta.get("is_fbcode", False))

        remote_cache = _create_cache(
            self.configs_hash,
            backend_hash,
            is_fbcode,
            "FbRemoteAutotuneCache",
            "RemoteAutotuneCache",
            "autotune-best-config-v2",
        )
        if not remote_cache:
            return

        # we already sha256 hash the source contents
        remote_cache_key = os.path.basename(filename)
        self.remote_cache = (remote_cache, remote_cache_key)

    # Save the config in the caches
    def save(
        self, config: Config, time_taken_ns: int, found_by_coordesc: bool = False
    ) -> None:
        data = {
            **config.kwargs,
            "num_warps": config.num_warps,
            "num_stages": config.num_stages,
            "configs_hash": self.configs_hash,
            "found_by_coordesc": found_by_coordesc,
            "time_taken_ms": time_taken_ns // 1000000,  # Convert from NS to MS
        }

        if local_cache := self.local_cache:
            cache, key = local_cache
            cache.put(key, data)
            AutotuneCacheBundler.put(key, data)

            if log.isEnabledFor(logging.DEBUG):
                type_str = "coordesc" if found_by_coordesc else "heuristic"
                log.debug("Save %s tuning result to %s", type_str, key)

        if remote_cache := self.remote_cache:
            cache, key = remote_cache
            cache.put(key, data)


class AutotuneCacheBundler:
    """
    Caches a set of LocalAutotuneCacheBackend entries together in a single
    cache.
    """

    # The current global AutotuneCacheBundler
    _CURRENT: Optional[AutotuneCacheBundler] = None

    _cache: RemoteCache[JsonDataTy]

    # All known entries from LocalAutotuneCache.put()
    _entries: Dict[str, JsonDataTy]

    @classmethod
    def begin_compile(
        cls,
        code_hash: str,
    ) -> None:
        #print("*** AutotuneCacheBundler.begin_compile")
        # traceback.print_stack()

        if cls._CURRENT is not None:
            # We already saw a begin_compile() but never got the corresponding
            # end_compile(). Likely this compile didn't actually generate
            # autotune timings. Cancel the existing compile.
            cls.cancel_compile()

        if not cls._should_use_bundled_autotune_cache():
            return None

        cache = _create_cache(
            code_hash,
            cls._get_backend_hash(),
            cls._get_is_fbcode(),
            "FbRemoteBundledAutotuneCache",
            "RemoteBundledAutotuneCache",
            "bundled-autotune-best-configs-v1",
        )
        if not cache:
            return None

        # We're starting a compilation phase. We have a cache key for the code
        # we're compiling. We'll get the individual autotune bundles later (via
        # self.put()). For now create the AutotuneCacheBundler and try to load
        # from the cache.

        bundler = AutotuneCacheBundler(code_hash, cache)
        if not bundler._load_cache():
            # We couldn't load from the cache - so make it global. Later on if
            # we add async load support then set _CURRENT here and clear it in
            # `sync()` if the cache was loaded.
            cls._CURRENT = bundler

    @classmethod
    def cancel_compile(cls) -> None:
        cls._CURRENT = None

    @classmethod
    def end_compile(cls) -> None:
        # This is called after the compile when all local caches have been filled.
        #print("*** AutotuneCacheBundler.end_compile")
        #traceback.print_stack()
        if cur := cls._CURRENT:
            cls._CURRENT = None
            cur._end_compile()

    def _end_compile(self) -> None:
        # TODO: Do we need to compute time_taken_ms and encode that somehow?
        self._cache.put("", self._entries)

    @staticmethod
    def sync() -> None:
        #print("*** AutotuneCacheBundler.sync")
        # We don't currently use this - but we could async load starting at
        # `begin_compile` and wait for the load to be finished here.
        pass

    @classmethod
    def put(cls, filename: str, data: JsonDataTy) -> None:
        #print(f"*** AutotuneCacheBundler.put({filename})")
        #traceback.print_stack()
        if cur := AutotuneCacheBundler._CURRENT:
            cur._put(filename, data)

    def _put(self, filename: str, data: JsonDataTy) -> None:
        # Do we need to worry about duplicates? We only have a single local fs
        # entry - so probably not.
        self._entries[filename] = data

    def __init__(self, code_hash: str, cache: RemoteCache[JsonDataTy]) -> None:
        self._cache = cache
        self._entries = {}

    @classmethod
    def _should_use_bundled_autotune_cache(cls):
        # TODO: Do we need to worry about non-runtime?
        from .. import config

        # The bundled autotune cache is only available if you've also got local
        # caching enabled.
        if (autotune_local_cache := config.autotune_local_cache) == False:
            return False

        # Check if the we're enabled via config
        if (bundled_autotune_cache := config.bundled_autotune_cache) is not None:
            return bool(bundled_autotune_cache)

        if not cls._get_is_fbcode():
            return False

        # TODO: Should this be the same constant as the autotune cache or a different one?
        try:
            from torch._inductor.fb.remote_cache import REMOTE_CACHE_VERSION
        except ModuleNotFoundError:
            return False

        # TODO: Should this be the same JK as the autotune cache or a different one?
        jk = torch._utils_internal.justknobs_getval_int(
            "pytorch/remote_cache:autotune_memcache_version"
        )
        return REMOTE_CACHE_VERSION >= jk

    def _load_cache(self) -> bool:
        # The single key is defined on construction of the cache.
        entries = self._cache.get("")
        if entries is None or not isinstance(entries, dict):
            # We couldn't load the cache - so mark _entries as non-None so we
            # store local cache values.
            return False

        # Go through the entries we got from the cache and save them locally.
        for filename, data in entries.items():
            local_cache = LocalAutotuneCache()
            local_cache.put(filename, data)

        return True

    @staticmethod
    def _get_is_fbcode() -> bool:
        # TODO: Do we need to worry about non-runtime?
        from .. import config

        return config.is_fbcode()

    @staticmethod
    def _get_backend_hash() -> str:
        # TODO: Do we need to worry about non-runtime?
        return torch.utils._triton.triton_hash_with_backend()


def _should_use_remote_autotune_cache(inductor_meta: Dict[str, object]) -> bool:
    if (config := inductor_meta.get("autotune_remote_cache")) is not None:
        return bool(config)
    if not inductor_meta.get("is_fbcode"):
        return False
    if torch._utils_internal.is_fb_unit_test():
        return False
    if inductor_meta.get("is_hip"):
        return False

    try:
        from torch._inductor.fb.remote_cache import REMOTE_CACHE_VERSION
    except ModuleNotFoundError:
        return False

    return REMOTE_CACHE_VERSION >= torch._utils_internal.justknobs_getval_int(
        "pytorch/remote_cache:autotune_memcache_version"
    )


def _load_cached_autotuning(
    best_config: Dict[str, JsonDataTy],
    configs_hash: str,
    configs: List[Config],
    inductor_meta: Dict[str, object],
) -> Optional[Config]:
    if best_config is None:
        return None
    if best_config.pop("configs_hash", None) != configs_hash:
        return None

    # Remove time taken for comparison
    best_config.pop("time_taken_ms", None)

    if inductor_meta.get("coordinate_descent_tuning") and best_config.pop(
        "found_by_coordesc", False
    ):
        num_warps = best_config.pop("num_warps")
        num_stages = best_config.pop("num_stages")
        triton_config = Config(best_config, num_warps=num_warps, num_stages=num_stages)
        triton_config.found_by_coordesc = True
        return triton_config

    matching_configs = [
        cfg
        for cfg in configs
        if all(val == best_config.get(key) for key, val in cfg.kwargs.items())
        and cfg.num_warps == best_config.get("num_warps")
        and cfg.num_stages == best_config.get("num_stages")
    ]
    if len(matching_configs) != 1:
        return None

    return matching_configs[0]


def _create_cache(
    key: str,
    backend_hash: str,
    is_fbcode: bool,
    fb_cache_cls: str,
    oss_cache_cls: str,
    salt: str,
) -> Optional[RemoteCache[JsonDataTy]]:
    key = backend_hash + key + salt
    key = hashlib.sha256(key.encode("utf-8")).hexdigest()

    try:
        if is_fbcode:
            import torch._inductor.fb.remote_cache

            cache_cls = getattr(torch._inductor.fb.remote_cache, fb_cache_cls)
            return cache_cls(key)
        else:
            import torch._inductor.remote_cache

            cache_cls = getattr(torch._inductor.remote_cache, oss_cache_cls)
            return cache_cls(key)
    except Exception:
        log.warning("Unable to create a remote cache", exc_info=True)
        return None


class _LocalAutotuneCacheBackend(RemoteCacheBackend[bytes]):
    @override
    def get(self, key: str) -> Optional[bytes]:
        try:
            with open(key, "rb") as fd:
                return fd.read()
        except FileNotFoundError:
            return None

    @override
    def put(self, key: str, data: bytes) -> None:
        with open(key, "wb") as fd:
            fd.write(data)


class LocalAutotuneCache(RemoteCache[JsonDataTy]):
    def __init__(self) -> None:
        backend = _LocalAutotuneCacheBackend()
        serde = RemoteCacheJsonSerde()
        super().__init__(backend, serde)

    @override
    def _backend_get(self, key: str) -> JsonDataTy:
        #print(f"*** LocalAutotuneCache.backend_get({key})")
        AutotuneCacheBundler.sync()
        return super()._backend_get(key)

    @override
    def _backend_put(self, key: str, data: JsonDataTy) -> None:
        #print(f"*** LocalAutotuneCache.backend_put({key})")
        AutotuneCacheBundler.put(key, data)
        super()._backend_put(key, data)
