#pragma once

// OK qq helper pour faciliter les accés au tenseur.
#include <cstddef>

// - les i se suivent
inline std::size_t pos2D(std::size_t l, std::size_t i, std::size_t j) {
    return i+l*j;
}
// si contigue
template<bool ROW_ORDER=true>
inline std::size_t pos2D(std::size_t M, std::size_t N, std::size_t i, std::size_t j) {
    if constexpr(ROW_ORDER) {
        return i+j*M;
    } else {
        return j+i*N;
    }
}

enum class TYPE_BLOC {
    PERFECT,
    FIN,
    NORMAL
};
template<std::size_t K1, std::size_t N0, std::size_t K0=1, TYPE_BLOC BT=TYPE_BLOC::PERFECT>
inline std::size_t posBloc2D(std::size_t K, std::size_t N, std::size_t k, std::size_t i) {
    // X(k,i)
    // X[K/K1][~N/N0][K1/K0][N0][K0]
    static_assert(K1%K0 == 0);
    // ici: K%K1 == 0 && N%N0 == 0  => voir pour le cas ou N%N0 != 0: il y a 1 "colone" plus petit a la fin
    if constexpr (BT==TYPE_BLOC::PERFECT) {
        // ici: K%K1 == 0 && N%N0 == 0
        return k%K0 + K0*(i%N0) + (N0*K0)*((k%K1)/K0) + K1*N0*(i/N0) + N*K1*(k/K1);
    } else
    if constexpr (BT==TYPE_BLOC::NORMAL) {
        // K%K1 == 0 && N%N0 != 0
        //const std::size_t NK0 = (N%N0)*K0;
        // bool fin = i>((N/N0)*N0);
        //return k%K0 + K0*(i%N0) + (fin?NK0:(N0*K0))*((k%K1)/K0) + K1*N0*(i/N0) + N*K1*(k/K1);
        bool fin = i>(N-(N%N0));
        return k%K0 + K0*(i%N0) + ((fin?(N%N0):N0)*K0)*((k%K1)/K0) + K1*N0*(i/N0) + N*K1*(k/K1);
    } else
    if constexpr (BT==TYPE_BLOC::FIN) {
        // K%K1 == 0 && N%N0 != 0 && on est dans les dernieres colones.
        return k%K0 + K0*(i%N0) + ((N%N0)*K0)*((k%K1)/K0) + K1*N0*(N/N0) + N*K1*(k/K1);
    }
}

//inline std::size_t blocPos2D(std::size_t _k0, std::size_t _i0, std::size_t _k1, std::size_t _i1, std::size_t K, std::size_t N) {
template<std::size_t K1, std::size_t N0, std::size_t K0=1, TYPE_BLOC BT=TYPE_BLOC::PERFECT>
inline std::size_t posBloc2D(std::size_t K, std::size_t N, std::size_t _k2, std::size_t _i1, std::size_t _k1, std::size_t _i0, std::size_t _k0=0) {
    // X[K/K1][~N/N0][K1/K0][N0][K0]
    // X(k0,i0,k1,i1,k2)
    static_assert(K1%K0 == 0);
    // ici: K%K1 == 0 && N%N0 == 0  => voir pour le cas ou N%N0 != 0: il y a 1 "colone" plus petit a la fin
    if constexpr (BT==TYPE_BLOC::PERFECT) {
        // ici: K%K1 == 0 && N%N0 == 0
        return _k0 + K0*_i0 + (N0*K0)*_k1 + K1*N0*_i1 + N*K1*_k2;
    } else
    if constexpr (BT==TYPE_BLOC::NORMAL) {
        // K%K1 == 0 && N%N0 != 0
        bool fin = _i1>(N/N0);
        return _k0 + K0*_i0 + ((fin?(N%N0):N0)*K0)*_k1 + K1*N0*_i1 + N*K1*_k2;
    } else
    if constexpr (BT==TYPE_BLOC::FIN) {
        // K%K1 == 0 && N%N0 != 0 et on est dans les dernieres colones.
        // && _i1>(N/N0)
        return _k0 + K0*_i0 + ((N%N0)*K0)*_k1 + K1*N0*(N/N0) + N*K1*_k2;
    }
}
