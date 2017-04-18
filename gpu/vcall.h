#pragma once

#include <cassert>

// Base Class
#define VCALL_DECL(_class, ret, name, ...) \
    ret (*_vptr_##name)(const _class * _this, ##__VA_ARGS__)
#define VCALL(name, ...) \
    (assert(_vptr_##name != nullptr), (*_vptr_##name)(this, ##__VA_ARGS__))

// Derived Class
#define VCALL_INIT(name) _vptr_##name = &_impl_##name
#define VCALL_IMPL(_class, ret, name, ...) \
    static ret _impl_##name(const _class * _this, ##__VA_ARGS__)
