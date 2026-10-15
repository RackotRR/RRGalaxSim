#pragma once
#include "co_types.h"

enum class BoundaryType : int {
    ZeroDirichlet     = 0, // старое поведение: phi = 0
    Periodic          = 1, // периодические
    MonopoleDirichlet = 2, // phi = -GM/r на границе
    MonopoleRadiative = 3, // монополь + условие излучения для отклонения
};

/// @brief Параметры граничных условий
struct BoundaryParams {
    BoundaryType type;

    /// @brief Полная масса и центр масс (для Monopole*)
    real mass;
    real3 center_of_mass;

    /// @brief Защита от r -> 0 (центр масс вне области или у границы)
    real r_min;

    /// @brief Средняя плотность (для Periodic) — заполняется на хосте
    real rho_mean;

    /// @brief 1 — оставить поглощающий слой в периодическом режиме
    int periodic_keep_layer;
};