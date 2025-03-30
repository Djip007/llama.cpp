#pragma once

#include "ggml-impl.h"
#include "types.h"

//##############################################################################################
// les methodes a instancier pour l'op mulmat de l' IGPU
namespace ggml::backend::igpu::op_mul_mat {
    // @ voir pour ca...
    // un template avec le "meme" proto que les mulmat ?
    //  ou on fait ca dans le supported?
    inline bool init_caches() {
        // @ voir...
        //if (B_cache.ensure_size(BLOC_K1 * 1024)) { // @ optimiser la taille de N < 768 * 16
        //    IGPU_TRACE("B_cache[" << BLOC_K1 << ", " << 1024 <<"]");
        //}
        return true;
    }

    //===========================================================================================================
    // l'interface du kernel...
    // pack des poids (A)
    template<typename TIN, typename TOUT>
    bool repack(const TIN* A, std::size_t la, TOUT* bloc, std::size_t M, std::size_t K);

    template<typename TA, typename TB, typename TC>
    bool supported(const ggml_tensor& A, const ggml_tensor& B, const ggml_tensor& C);

    // compute
    template<typename TA, typename TB, typename TC>
    bool compute(const TA* A, const TB* B, TC* C,
            std::size_t M, std::size_t N, std::size_t K,
            std::size_t la, std::size_t lb, std::size_t lc);
}
