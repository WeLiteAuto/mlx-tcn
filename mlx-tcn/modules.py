from typing import Iterable, Iterator, List, Optional, Sequence, Union

import mlx.nn as nn


class ModuleList(nn.Module):
    """A lightweight MLX equivalent of PyTorch's nn.ModuleList.

    Modules stored in the list are registered so their parameters appear in
    optimizers, state dictionaries, and recursive visitors.
    """

    def __init__(self, modules: Optional[Iterable[nn.Module]] = None):
        super().__init__()
        self._modules_list: List[nn.Module] = []
        self._module_keys: List[str] = []
        if modules is not None:
            self.extend(modules)

    def _check_module(self, module: nn.Module) -> None:
        if not isinstance(module, nn.Module):
            raise TypeError("ModuleList can only store instances of nn.Module.")

    def _sync_attributes(self) -> None:
        """Ensure list order matches the module attributes stored on self."""
        for key in list(self._module_keys):
            if hasattr(self, key):
                delattr(self, key)
        self._module_keys = []

        for idx, module in enumerate(self._modules_list):
            key = str(idx)
            self._module_keys.append(key)
            setattr(self, key, module)

    def __len__(self) -> int:
        return len(self._modules_list)

    def __iter__(self) -> Iterator[nn.Module]:
        return iter(self._modules_list)

    def __getitem__(self, index: Union[int, slice]) -> Union[nn.Module, "ModuleList"]:
        if isinstance(index, slice):
            return ModuleList(self._modules_list[index])
        return self._modules_list[index]

    def __setitem__(self, index: Union[int, slice], module: Union[nn.Module, Iterable[nn.Module]]) -> None:
        if isinstance(index, slice):
            modules = list(module) if isinstance(module, Iterable) else None  # type: ignore[assignment]
            if modules is None:
                raise TypeError("Assigning to a slice requires an iterable of nn.Module instances.")
            for mod in modules:
                self._check_module(mod)
            self._modules_list[index] = modules  # type: ignore[index]
        else:
            self._check_module(module)  # type: ignore[arg-type]
            self._modules_list[index] = module  # type: ignore[index]
        self._sync_attributes()

    def __repr__(self) -> str:
        child_lines = []
        for module in self._modules_list:
            child_lines.append(f"  ({len(child_lines)}): {module!r}")
        lines = ["ModuleList("]
        lines.extend(child_lines)
        lines.append(")")
        return "\n".join(lines)

    def append(self, module: nn.Module) -> "ModuleList":
        """Add a single module to the end of the list."""
        self._check_module(module)
        self._modules_list.append(module)
        self._sync_attributes()
        return self

    def extend(self, modules: Iterable[nn.Module]) -> "ModuleList":
        """Append modules from an iterable to the list."""
        for module in modules:
            self._check_module(module)
            self._modules_list.append(module)
        self._sync_attributes()
        return self

    def insert(self, index: int, module: nn.Module) -> None:
        """Insert a module at the specified position."""
        self._check_module(module)
        self._modules_list.insert(index, module)
        self._sync_attributes()

    def pop(self, index: int = -1) -> nn.Module:
        """Remove and return a module from the list."""
        module = self._modules_list.pop(index)
        self._sync_attributes()
        return module

    def clear(self) -> None:
        """Remove all modules from the container."""
        self._modules_list.clear()
        self._sync_attributes()

    def __add__(self, other: Sequence[nn.Module]) -> "ModuleList":
        if not isinstance(other, Sequence):
            raise TypeError("ModuleList can only be concatenated with a sequence of nn.Module.")
        combined = ModuleList(self._modules_list)
        combined.extend(other)
        return combined

    def __iadd__(self, other: Iterable[nn.Module]) -> "ModuleList":
        return self.extend(other)
