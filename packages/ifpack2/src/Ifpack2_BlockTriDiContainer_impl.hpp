// @HEADER
// *****************************************************************************
//       Ifpack2: Templated Object-Oriented Algebraic Preconditioner Package
//
// Copyright 2009 NTESS and the Ifpack2 contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef IFPACK2_BLOCKTRIDICONTAINER_IMPL_HPP
#define IFPACK2_BLOCKTRIDICONTAINER_IMPL_HPP

//#define IFPACK2_BLOCKTRIDICONTAINER_WRITE_MM
//#define IFPACK2_BLOCKTRIDICONTAINER_USE_PRINTF

#include <Teuchos_Details_MpiTypeTraits.hpp>

#include <Tpetra_Details_extractMpiCommFromTeuchos.hpp>
#include <Tpetra_Distributor.hpp>
#include <Tpetra_BlockMultiVector.hpp>

#include <Kokkos_ArithTraits.hpp>
#include <KokkosBatched_Util.hpp>
#include <KokkosBatched_Vector.hpp>
#include <KokkosBatched_Copy_Decl.hpp>
#include <KokkosBatched_Copy_Impl.hpp>
#include <KokkosBatched_AddRadial_Decl.hpp>
#include <KokkosBatched_AddRadial_Impl.hpp>
#include <KokkosBatched_SetIdentity_Decl.hpp>
#include <KokkosBatched_SetIdentity_Impl.hpp>
#include <KokkosBatched_Gemm_Decl.hpp>
#include <KokkosBatched_Gemm_Serial_Impl.hpp>
#include <KokkosBatched_Gemm_Team_Impl.hpp>
#include <KokkosBatched_Gemv_Decl.hpp>
#include <KokkosBatched_Gemv_Team_Impl.hpp>
#include <KokkosBatched_Trsm_Decl.hpp>
#include <KokkosBatched_Trsm_Serial_Impl.hpp>
#include <KokkosBatched_Trsm_Team_Impl.hpp>
#include <KokkosBatched_Trsv_Decl.hpp>
#include <KokkosBatched_Trsv_Serial_Impl.hpp>
#include <KokkosBatched_Trsv_Team_Impl.hpp>
#include <KokkosBatched_LU_Decl.hpp>
#include <KokkosBatched_LU_Serial_Impl.hpp>
#include <KokkosBatched_LU_Team_Impl.hpp>

#include <KokkosBlas1_nrm1.hpp>
#include <KokkosBlas1_nrm2.hpp>

#include <memory>

#include "Ifpack2_BlockHelper.hpp"
#include "Tpetra_Details_residual.hpp"

//#include <KokkosBlas2_gemv.hpp>

// need to interface this into cmake variable (or only use this flag when it is necessary)
//#define IFPACK2_BLOCKTRIDICONTAINER_ENABLE_PROFILE
//#undef  IFPACK2_BLOCKTRIDICONTAINER_ENABLE_PROFILE
#if defined(KOKKOS_ENABLE_CUDA) && defined(IFPACK2_BLOCKTRIDICONTAINER_ENABLE_PROFILE)
#include "cuda_profiler_api.h"
#endif

// I am not 100% sure about the mpi 3 on cuda
#if MPI_VERSION >= 3
#define IFPACK2_BLOCKTRIDICONTAINER_USE_MPI_3
#endif

// ::: Experiments :::
// define either pinned memory or cudamemory for mpi
// if both macros are disabled, it will use tpetra memory space which is uvm space for cuda
// if defined, this use pinned memory instead of device pointer
// by default, we enable pinned memory
#define IFPACK2_BLOCKTRIDICONTAINER_USE_PINNED_MEMORY_FOR_MPI
//#define IFPACK2_BLOCKTRIDICONTAINER_USE_CUDA_MEMORY_FOR_MPI

// if defined, all views are allocated on cuda space intead of cuda uvm space
#define IFPACK2_BLOCKTRIDICONTAINER_USE_CUDA_SPACE

// if defined, btdm_scalar_type is used (if impl_scala_type is double, btdm_scalar_type is float)
#if defined(HAVE_IFPACK2_BLOCKTRIDICONTAINER_SMALL_SCALAR)
#define IFPACK2_BLOCKTRIDICONTAINER_USE_SMALL_SCALAR_FOR_BLOCKTRIDIAG
#endif

// if defined, it uses multiple execution spaces
#define IFPACK2_BLOCKTRIDICONTAINER_USE_EXEC_SPACE_INSTANCES

namespace Ifpack2 {

  namespace BlockTriDiContainerDetails {

    namespace KB = KokkosBatched;

    ///
    /// view decorators for unmanaged and const memory
    ///
    using do_not_initialize_tag = Kokkos::ViewAllocateWithoutInitializing;

    template <typename MemoryTraitsType, Kokkos::MemoryTraitsFlags flag>
    using MemoryTraits = Kokkos::MemoryTraits<MemoryTraitsType::is_unmanaged |
                                              MemoryTraitsType::is_random_access |
                                              flag>;

    template <typename ViewType>
    using Unmanaged = Kokkos::View<typename ViewType::data_type,
                                   typename ViewType::array_layout,
                                   typename ViewType::device_type,
                                  MemoryTraits<typename ViewType::memory_traits,Kokkos::Unmanaged> >;
    template <typename ViewType>
    using Atomic = Kokkos::View<typename ViewType::data_type,
                                typename ViewType::array_layout,
                                typename ViewType::device_type,
                                MemoryTraits<typename ViewType::memory_traits,Kokkos::Atomic> >;
    template <typename ViewType>
    using Const = Kokkos::View<typename ViewType::const_data_type,
                               typename ViewType::array_layout,
                               typename ViewType::device_type,
                               typename ViewType::memory_traits>;
    template <typename ViewType>
    using ConstUnmanaged = Const<Unmanaged<ViewType> >;

    template <typename ViewType>
    using AtomicUnmanaged = Atomic<Unmanaged<ViewType> >;

    template <typename ViewType>
    using Unmanaged = Kokkos::View<typename ViewType::data_type,
                                   typename ViewType::array_layout,
                                   typename ViewType::device_type,
                                   MemoryTraits<typename ViewType::memory_traits,Kokkos::Unmanaged> >;


    template <typename ViewType>
    using Scratch = Kokkos::View<typename ViewType::data_type,
                                 typename ViewType::array_layout,
                                 typename ViewType::execution_space::scratch_memory_space,
                                 MemoryTraits<typename ViewType::memory_traits, Kokkos::Unmanaged> >;

    ///
    /// block tridiag scalar type
    ///
    template<typename T> struct BlockTridiagScalarType { typedef T type; };
#if defined(IFPACK2_BLOCKTRIDICONTAINER_USE_SMALL_SCALAR_FOR_BLOCKTRIDIAG)
    template<> struct BlockTridiagScalarType<double> { typedef float type; };
    //template<> struct SmallScalarType<Kokkos::complex<double> > { typedef Kokkos::complex<float> type; };
#endif

#if defined(KOKKOS_ENABLE_CUDA) && defined(IFPACK2_BLOCKTRIDICONTAINER_ENABLE_PROFILE)
#define IFPACK2_BLOCKTRIDICONTAINER_PROFILER_REGION_BEGIN \
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaProfilerStart());

#define IFPACK2_BLOCKTRIDICONTAINER_PROFILER_REGION_END \
    { KOKKOS_IMPL_CUDA_SAFE_CALL( cudaProfilerStop() ); }
#else
    /// later put vtune profiler region
#define IFPACK2_BLOCKTRIDICONTAINER_PROFILER_REGION_BEGIN
#define IFPACK2_BLOCKTRIDICONTAINER_PROFILER_REGION_END
#endif

    template<typename local_ordinal_type>
    local_ordinal_type costTRSM(const local_ordinal_type block_size) {
      return block_size*block_size;
    }

    template<typename local_ordinal_type>
    local_ordinal_type costGEMV(const local_ordinal_type block_size) {
      return 2*block_size*block_size;
    }

    template<typename local_ordinal_type>
    local_ordinal_type costTriDiagSolve(const local_ordinal_type subline_length, const local_ordinal_type block_size) {
      return 2 * subline_length * costTRSM(block_size) + 2 * (subline_length-1) * costGEMV(block_size);
    }

    template<typename local_ordinal_type>
    local_ordinal_type costSolveSchur(const local_ordinal_type num_parts,
                                      const local_ordinal_type num_teams,
                                      const local_ordinal_type line_length,
                                      const local_ordinal_type block_size,
                                      const local_ordinal_type n_subparts_per_part) {
      const local_ordinal_type subline_length = ceil(double(line_length - (n_subparts_per_part-1) * 2) / n_subparts_per_part);
      if (subline_length < 1) {
        return INT_MAX;
      }

      const local_ordinal_type p_n_lines = ceil(double(num_parts)/num_teams);
      const local_ordinal_type p_n_sublines = ceil(double(n_subparts_per_part)*num_parts/num_teams);
      const local_ordinal_type p_n_sublines_2 = ceil(double(n_subparts_per_part-1)*num_parts/num_teams);

      const local_ordinal_type p_costApplyE = p_n_sublines_2 * subline_length * 2 * costGEMV(block_size);
      const local_ordinal_type p_costApplyS = p_n_lines * costTriDiagSolve((n_subparts_per_part-1)*2,block_size);
      const local_ordinal_type p_costApplyAinv = p_n_sublines * costTriDiagSolve(subline_length,block_size);
      const local_ordinal_type p_costApplyC = p_n_sublines_2 * 2 * costGEMV(block_size);

      if (n_subparts_per_part == 1) {
        return p_costApplyAinv;
      }
      return p_costApplyE + p_costApplyS + p_costApplyAinv + p_costApplyC;
    }

    template<typename local_ordinal_type>
    local_ordinal_type getAutomaticNSubparts(const local_ordinal_type num_parts,
                                             const local_ordinal_type num_teams,
                                             const local_ordinal_type line_length,
                                             const local_ordinal_type block_size) {
      local_ordinal_type n_subparts_per_part_0 = 1;
      local_ordinal_type flop_0 = costSolveSchur(num_parts, num_teams, line_length, block_size, n_subparts_per_part_0);
      local_ordinal_type flop_1 = costSolveSchur(num_parts, num_teams, line_length, block_size, n_subparts_per_part_0+1);
      while (flop_0 > flop_1) {
        flop_0 = flop_1;
        flop_1 = costSolveSchur(num_parts, num_teams, line_length, block_size, (++n_subparts_per_part_0)+1);
      }
      return n_subparts_per_part_0;
    }

    template<typename ArgActiveExecutionMemorySpace>
    struct SolveTridiagsDefaultModeAndAlgo;

    ///
    /// setup part interface using the container partitions array
    ///
    template<typename MatrixType>
    BlockHelperDetails::PartInterface<MatrixType>
    createPartInterface(const Teuchos::RCP<const typename BlockHelperDetails::ImplType<MatrixType>::tpetra_row_matrix_type> &A,
                        const Teuchos::RCP<const typename BlockHelperDetails::ImplType<MatrixType>::tpetra_crs_graph_type> &G,
                        const Teuchos::Array<Teuchos::Array<typename BlockHelperDetails::ImplType<MatrixType>::local_ordinal_type> > &partitions,
                        const typename BlockHelperDetails::ImplType<MatrixType>::local_ordinal_type n_subparts_per_part_in) {
      IFPACK2_BLOCKHELPER_TIMER("createPartInterface", createPartInterface);
      using impl_type = BlockHelperDetails::ImplType<MatrixType>;
      using local_ordinal_type = typename impl_type::local_ordinal_type;
      using local_ordinal_type_1d_view = typename impl_type::local_ordinal_type_1d_view;
      using local_ordinal_type_2d_view = typename impl_type::local_ordinal_type_2d_view;
      using size_type = typename impl_type::size_type;

      auto bA = Teuchos::rcp_dynamic_cast<const typename BlockHelperDetails::ImplType<MatrixType>::tpetra_block_crs_matrix_type>(A);

      TEUCHOS_ASSERT(!bA.is_null() || G->getLocalNumRows() != 0);
      const local_ordinal_type blocksize = bA.is_null() ? A->getLocalNumRows() / G->getLocalNumRows() : A->getBlockSize();
      constexpr int vector_length = impl_type::vector_length;
      constexpr int internal_vector_length = impl_type::internal_vector_length;

      const auto comm = A->getRowMap()->getComm();

      BlockHelperDetails::PartInterface<MatrixType> interf;

      const bool jacobi = partitions.size() == 0;
      const local_ordinal_type A_n_lclrows = G->getLocalNumRows();
      const local_ordinal_type nparts = jacobi ? A_n_lclrows : partitions.size();

      typedef std::pair<local_ordinal_type,local_ordinal_type> size_idx_pair_type;
      std::vector<size_idx_pair_type> partsz(nparts);

      if (!jacobi) {
        for (local_ordinal_type i=0;i<nparts;++i)
          partsz[i] = size_idx_pair_type(partitions[i].size(), i);
        std::sort(partsz.begin(), partsz.end(),
                  [] (const size_idx_pair_type& x, const size_idx_pair_type& y) {
                    return x.first > y.first;
                  });
      }

      local_ordinal_type n_subparts_per_part;
      if (n_subparts_per_part_in == -1) {
        // If the number of subparts is set to -1, the user let the algorithm
        // decides the value automatically
        using execution_space = typename impl_type::execution_space;

        const int line_length = partsz[0].first;

        const local_ordinal_type team_size = 
          SolveTridiagsDefaultModeAndAlgo<typename execution_space::memory_space>::
          recommended_team_size(blocksize, vector_length, internal_vector_length);

        const local_ordinal_type num_teams = std::max(1, execution_space().concurrency() / (team_size * vector_length));

        n_subparts_per_part = getAutomaticNSubparts(nparts, num_teams, line_length, blocksize);

#ifdef IFPACK2_BLOCKTRIDICONTAINER_USE_PRINTF
        printf("Automatically chosen n_subparts_per_part = %d for nparts = %d, num_teams = %d, team_size = %d, line_length = %d, and blocksize = %d;\n", n_subparts_per_part, nparts, num_teams, team_size, line_length, blocksize);
#endif
      }
      else {
        n_subparts_per_part = n_subparts_per_part_in;
      }

      // Total number of sub lines:
      const local_ordinal_type n_sub_parts = nparts * n_subparts_per_part;
      // Total number of sub lines + the Schur complement blocks.
      // For a given live 2 sub lines implies one Schur complement, 3 sub lines implies two Schur complements etc.
      const local_ordinal_type n_sub_parts_and_schur = n_sub_parts + nparts * (n_subparts_per_part-1);

#if defined(BLOCKTRIDICONTAINER_DEBUG)
      local_ordinal_type nrows = 0;
      if (jacobi)
        nrows = nparts;
      else
        for (local_ordinal_type i=0;i<nparts;++i) nrows += partitions[i].size();

      TEUCHOS_TEST_FOR_EXCEPT_MSG
        (nrows != A_n_lclrows, BlockHelperDetails::get_msg_prefix(comm) << "The #rows implied by the local partition is not "
         << "the same as getLocalNumRows: " << nrows << " vs " << A_n_lclrows);
#endif

      // permutation vector
      std::vector<local_ordinal_type> p;
      if (jacobi) {
        interf.max_partsz = 1;
        interf.max_subpartsz = 0;
        interf.n_subparts_per_part = 1;
        interf.nparts = nparts;
      } else {
        // reorder parts to maximize simd packing efficiency
        p.resize(nparts);

        for (local_ordinal_type i=0;i<nparts;++i)
          p[i] = partsz[i].second;

        interf.max_partsz = partsz[0].first;

        constexpr local_ordinal_type connection_length = 2;
        const local_ordinal_type sub_line_length = (interf.max_partsz - (n_subparts_per_part - 1) * connection_length) / n_subparts_per_part;
        const local_ordinal_type last_sub_line_length = interf.max_partsz - (n_subparts_per_part - 1) * (connection_length + sub_line_length);

        interf.max_subpartsz = (sub_line_length > last_sub_line_length) ? sub_line_length : last_sub_line_length;
        interf.n_subparts_per_part = n_subparts_per_part;
        interf.nparts = nparts;
      }

      // allocate parts
      interf.partptr = local_ordinal_type_1d_view(do_not_initialize_tag("partptr"), nparts + 1);
      interf.lclrow = local_ordinal_type_1d_view(do_not_initialize_tag("lclrow"), A_n_lclrows);
      interf.part2rowidx0 = local_ordinal_type_1d_view(do_not_initialize_tag("part2rowidx0"), nparts + 1);
      interf.part2packrowidx0 = local_ordinal_type_1d_view(do_not_initialize_tag("part2packrowidx0"), nparts + 1);
      interf.rowidx2part = local_ordinal_type_1d_view(do_not_initialize_tag("rowidx2part"), A_n_lclrows);

      interf.part2rowidx0_sub = local_ordinal_type_1d_view(do_not_initialize_tag("part2rowidx0_sub"), n_sub_parts_and_schur + 1);
      interf.part2packrowidx0_sub = local_ordinal_type_2d_view(do_not_initialize_tag("part2packrowidx0_sub"), nparts, 2 * n_subparts_per_part);
      interf.rowidx2part_sub = local_ordinal_type_1d_view(do_not_initialize_tag("rowidx2part"), A_n_lclrows);

      interf.partptr_sub = local_ordinal_type_2d_view(do_not_initialize_tag("partptr_sub"), n_sub_parts_and_schur, 2);

      // mirror to host and compute on host execution space
      const auto partptr = Kokkos::create_mirror_view(interf.partptr);
      const auto partptr_sub = Kokkos::create_mirror_view(interf.partptr_sub);
      
      const auto lclrow = Kokkos::create_mirror_view(interf.lclrow);
      const auto part2rowidx0 = Kokkos::create_mirror_view(interf.part2rowidx0);
      const auto part2packrowidx0 = Kokkos::create_mirror_view(interf.part2packrowidx0);
      const auto rowidx2part = Kokkos::create_mirror_view(interf.rowidx2part);

      const auto part2rowidx0_sub = Kokkos::create_mirror_view(interf.part2rowidx0_sub);
      const auto part2packrowidx0_sub = Kokkos::create_mirror_view(Kokkos::HostSpace(), interf.part2packrowidx0_sub);
      const auto rowidx2part_sub = Kokkos::create_mirror_view(interf.rowidx2part_sub);

      // Determine parts.
      interf.row_contiguous = true;
      partptr(0) = 0;
      part2rowidx0(0) = 0;
      part2packrowidx0(0) = 0;
      local_ordinal_type pack_nrows = 0;
      local_ordinal_type pack_nrows_sub = 0;
      if (jacobi) {
        IFPACK2_BLOCKHELPER_TIMER("compute part indices (Jacobi)", Jacobi);
        for (local_ordinal_type ip=0;ip<nparts;++ip) {
          constexpr local_ordinal_type ipnrows = 1;
          //assume No overlap.
          part2rowidx0(ip+1) = part2rowidx0(ip) + ipnrows;
          // Since parts are ordered in decreasing size, the size of the first
          // part in a pack is the size for all parts in the pack.
          if (ip % vector_length == 0) pack_nrows = ipnrows;
          part2packrowidx0(ip+1) = part2packrowidx0(ip) + ((ip+1) % vector_length == 0 || ip+1 == nparts ? pack_nrows : 0);
          const local_ordinal_type offset = partptr(ip);
          for (local_ordinal_type i=0;i<ipnrows;++i) {
            const auto lcl_row = ip;
            TEUCHOS_TEST_FOR_EXCEPT_MSG(lcl_row < 0 || lcl_row >= A_n_lclrows,
                BlockHelperDetails::get_msg_prefix(comm)
                << "partitions[" << p[ip] << "]["
                << i << "] = " << lcl_row
                << " but input matrix implies limits of [0, " << A_n_lclrows-1
                << "].");
            lclrow(offset+i) = lcl_row;
            rowidx2part(offset+i) = ip;
            if (interf.row_contiguous && offset+i > 0 && lclrow((offset+i)-1) + 1 != lcl_row)
              interf.row_contiguous = false;
          }
          partptr(ip+1) = offset + ipnrows;
        }
        part2rowidx0_sub(0) = 0;
        partptr_sub(0, 0) = 0;

        for (local_ordinal_type ip=0;ip<nparts;++ip) {
          constexpr local_ordinal_type ipnrows = 1;
          const local_ordinal_type full_line_length = partptr(ip+1) - partptr(ip);

          TEUCHOS_TEST_FOR_EXCEPTION
            (full_line_length != ipnrows, std::logic_error, 
            "In the part " << ip );  

          constexpr local_ordinal_type connection_length = 2;

          if (full_line_length < n_subparts_per_part + (n_subparts_per_part - 1) * connection_length )
              TEUCHOS_TEST_FOR_EXCEPTION
                (true, std::logic_error, 
                "The part " << ip << " is too short to use " << n_subparts_per_part << " sub parts.");            

          const local_ordinal_type sub_line_length = (full_line_length - (n_subparts_per_part - 1) * connection_length) / n_subparts_per_part;
          const local_ordinal_type last_sub_line_length = full_line_length - (n_subparts_per_part - 1) * (connection_length + sub_line_length);

          if (ip % vector_length == 0) pack_nrows_sub = ipnrows;

          for (local_ordinal_type local_sub_ip=0; local_sub_ip<n_subparts_per_part;++local_sub_ip) {
            const local_ordinal_type sub_ip = nparts*(2*local_sub_ip) + ip;
            const local_ordinal_type schur_ip = nparts*(2*local_sub_ip+1) + ip;
            if (local_sub_ip != n_subparts_per_part-1) {
              if (local_sub_ip != 0) {
                partptr_sub(sub_ip, 0) = partptr_sub(nparts*(2*local_sub_ip-1) + ip, 1);
              }
              else if (ip != 0) {
                partptr_sub(sub_ip, 0) = partptr_sub(nparts*2*(n_subparts_per_part-1) + ip - 1, 1);
              }
              partptr_sub(sub_ip, 1) = sub_line_length + partptr_sub(sub_ip, 0);
              partptr_sub(schur_ip, 0) = partptr_sub(sub_ip, 1);
              partptr_sub(schur_ip, 1) = connection_length + partptr_sub(schur_ip, 0);

              part2rowidx0_sub(sub_ip + 1) = part2rowidx0_sub(sub_ip) + sub_line_length;
              part2rowidx0_sub(sub_ip + 2) = part2rowidx0_sub(sub_ip + 1) + connection_length;

#ifdef IFPACK2_BLOCKTRIDICONTAINER_USE_PRINTF
              printf("Sub Part index = %d, first LID associated to the sub part = %d, sub part size = %d;\n", sub_ip, partptr_sub(ip, 2 * local_sub_ip), sub_line_length);
              printf("Sub Part index Schur = %d, first LID associated to the sub part = %d, sub part size = %d;\n", sub_ip + 1, partptr_sub(ip, 2 * local_sub_ip + 1), connection_length);
#endif
            }
            else {
              if (local_sub_ip != 0) {
                partptr_sub(sub_ip, 0) = partptr_sub(nparts*(2*local_sub_ip-1) + ip, 1);
              }
              else if (ip != 0) {
                partptr_sub(sub_ip, 0) = partptr_sub(nparts*2*(n_subparts_per_part-1) + ip - 1, 1);
              }
              partptr_sub(sub_ip, 1) = last_sub_line_length + partptr_sub(sub_ip, 0);

              part2rowidx0_sub(sub_ip + 1) = part2rowidx0_sub(sub_ip) + last_sub_line_length;

#ifdef IFPACK2_BLOCKTRIDICONTAINER_USE_PRINTF
              printf("Sub Part index = %d, first LID associated to the sub part = %d, sub part size = %d;\n", sub_ip, partptr_sub(ip, 2 * local_sub_ip), last_sub_line_length);
#endif
            }
          }
        }

#ifdef IFPACK2_BLOCKTRIDICONTAINER_WRITE_MM
        std::cout << "partptr_sub = " << std::endl;
        for (size_type i = 0; i < partptr_sub.extent(0); ++i) {
          for (size_type j = 0; j < partptr_sub.extent(1); ++j) {
            std::cout << partptr_sub(i,j) << " ";
          }
          std::cout << std::endl;
        }
        std::cout << "partptr_sub end" << std::endl;
#endif

        {
          local_ordinal_type npacks = ceil(float(nparts)/vector_length);

          local_ordinal_type ip_max = nparts > vector_length ? vector_length : nparts;
          for (local_ordinal_type ip=0;ip<ip_max;++ip) {
            part2packrowidx0_sub(ip, 0) = 0;
          }
          for (local_ordinal_type ipack=0;ipack<npacks;++ipack) {
            if (ipack != 0) {
              local_ordinal_type ip_min = ipack*vector_length;
              ip_max = nparts > (ipack+1)*vector_length ? (ipack+1)*vector_length : nparts;
              for (local_ordinal_type ip=ip_min;ip<ip_max;++ip) {
                part2packrowidx0_sub(ip, 0) = part2packrowidx0_sub(ip-vector_length, part2packrowidx0_sub.extent(1)-1);
              }
            }

            for (size_type local_sub_ip=0; local_sub_ip<part2packrowidx0_sub.extent(1)-1;++local_sub_ip) {
              local_ordinal_type ip_min = ipack*vector_length;
              ip_max = nparts > (ipack+1)*vector_length ? (ipack+1)*vector_length : nparts;

              const local_ordinal_type full_line_length = partptr(ip_min+1) - partptr(ip_min);

              constexpr local_ordinal_type connection_length = 2;      

              const local_ordinal_type sub_line_length = (full_line_length - (n_subparts_per_part - 1) * connection_length) / n_subparts_per_part;
              const local_ordinal_type last_sub_line_length = full_line_length - (n_subparts_per_part - 1) * (connection_length + sub_line_length);

              if (local_sub_ip % 2 == 0) pack_nrows_sub = sub_line_length;
              if (local_sub_ip % 2 == 1) pack_nrows_sub = connection_length;
              if (local_sub_ip == part2packrowidx0_sub.extent(1)-2) pack_nrows_sub = last_sub_line_length;

              part2packrowidx0_sub(ip_min, local_sub_ip + 1) = part2packrowidx0_sub(ip_min, local_sub_ip) + pack_nrows_sub;

              for (local_ordinal_type ip=ip_min+1;ip<ip_max;++ip) {
                part2packrowidx0_sub(ip, local_sub_ip + 1) = part2packrowidx0_sub(ip_min, local_sub_ip + 1);
              }
            }
          }

          Kokkos::deep_copy(interf.part2packrowidx0_sub, part2packrowidx0_sub);
        }        
        IFPACK2_BLOCKHELPER_TIMER_FENCE(typename BlockHelperDetails::ImplType<MatrixType>::execution_space)
      } else {
        IFPACK2_BLOCKHELPER_TIMER("compute part indices", indices);
        for (local_ordinal_type ip=0;ip<nparts;++ip) {
          const auto* part = &partitions[p[ip]];
          const local_ordinal_type ipnrows = part->size();
          TEUCHOS_ASSERT(ip == 0 || (ipnrows <= static_cast<local_ordinal_type>(partitions[p[ip-1]].size())));
          TEUCHOS_TEST_FOR_EXCEPT_MSG(ipnrows == 0,
                    BlockHelperDetails::get_msg_prefix(comm)
                    << "partition " << p[ip]
                    << " is empty, which is not allowed.");
          //assume No overlap.
          part2rowidx0(ip+1) = part2rowidx0(ip) + ipnrows;
          // Since parts are ordered in decreasing size, the size of the first
          // part in a pack is the size for all parts in the pack.
          if (ip % vector_length == 0) pack_nrows = ipnrows;
          part2packrowidx0(ip+1) = part2packrowidx0(ip) + ((ip+1) % vector_length == 0 || ip+1 == nparts ? pack_nrows : 0);
          const local_ordinal_type offset = partptr(ip);
          for (local_ordinal_type i=0;i<ipnrows;++i) {
            const auto lcl_row = (*part)[i];
            TEUCHOS_TEST_FOR_EXCEPT_MSG(lcl_row < 0 || lcl_row >= A_n_lclrows,
                BlockHelperDetails::get_msg_prefix(comm)
                << "partitions[" << p[ip] << "]["
                << i << "] = " << lcl_row
                << " but input matrix implies limits of [0, " << A_n_lclrows-1
                << "].");
            lclrow(offset+i) = lcl_row;
            rowidx2part(offset+i) = ip;
            if (interf.row_contiguous && offset+i > 0 && lclrow((offset+i)-1) + 1 != lcl_row)
              interf.row_contiguous = false;
          }
          partptr(ip+1) = offset + ipnrows;

#ifdef IFPACK2_BLOCKTRIDICONTAINER_USE_PRINTF
          printf("Part index = ip = %d, first LID associated to the part = partptr(ip) = offset = %d, part->size() = ipnrows = %d;\n", ip, offset, ipnrows);
          printf("partptr(%d+1) = %d\n", ip, partptr(ip+1));
#endif
        }

        part2rowidx0_sub(0) = 0;
        partptr_sub(0, 0) = 0;
        //const local_ordinal_type number_pack_per_sub_part = ceil(float(nparts)/vector_length);

        for (local_ordinal_type ip=0;ip<nparts;++ip) {
          const auto* part = &partitions[p[ip]];
          const local_ordinal_type ipnrows = part->size();
          const local_ordinal_type full_line_length = partptr(ip+1) - partptr(ip);

          TEUCHOS_TEST_FOR_EXCEPTION
            (full_line_length != ipnrows, std::logic_error, 
            "In the part " << ip );  

          constexpr local_ordinal_type connection_length = 2;

          if (full_line_length < n_subparts_per_part + (n_subparts_per_part - 1) * connection_length )
              TEUCHOS_TEST_FOR_EXCEPTION
                (true, std::logic_error, 
                "The part " << ip << " is too short to use " << n_subparts_per_part << " sub parts.");            

          const local_ordinal_type sub_line_length = (full_line_length - (n_subparts_per_part - 1) * connection_length) / n_subparts_per_part;
          const local_ordinal_type last_sub_line_length = full_line_length - (n_subparts_per_part - 1) * (connection_length + sub_line_length);

          if (ip % vector_length == 0) pack_nrows_sub = ipnrows;

          for (local_ordinal_type local_sub_ip=0; local_sub_ip<n_subparts_per_part;++local_sub_ip) {
            const local_ordinal_type sub_ip = nparts*(2*local_sub_ip) + ip;
            const local_ordinal_type schur_ip = nparts*(2*local_sub_ip+1) + ip;
            if (local_sub_ip != n_subparts_per_part-1) {
              if (local_sub_ip != 0) {
                partptr_sub(sub_ip, 0) = partptr_sub(nparts*(2*local_sub_ip-1) + ip, 1);
              }
              else if (ip != 0) {
                partptr_sub(sub_ip, 0) = partptr_sub(nparts*2*(n_subparts_per_part-1) + ip - 1, 1);
              }
              partptr_sub(sub_ip, 1) = sub_line_length + partptr_sub(sub_ip, 0);
              partptr_sub(schur_ip, 0) = partptr_sub(sub_ip, 1);
              partptr_sub(schur_ip, 1) = connection_length + partptr_sub(schur_ip, 0);

              part2rowidx0_sub(sub_ip + 1) = part2rowidx0_sub(sub_ip) + sub_line_length;
              part2rowidx0_sub(sub_ip + 2) = part2rowidx0_sub(sub_ip + 1) + connection_length;

#ifdef IFPACK2_BLOCKTRIDICONTAINER_USE_PRINTF
              printf("Sub Part index = %d, first LID associated to the sub part = %d, sub part size = %d;\n", sub_ip, partptr_sub(sub_ip, 0), sub_line_length);
              printf("Sub Part index Schur = %d, first LID associated to the sub part = %d, sub part size = %d;\n", sub_ip + 1, partptr_sub(ip, 2 * local_sub_ip + 1), connection_length);
#endif
            }
            else {
              if (local_sub_ip != 0) {
                partptr_sub(sub_ip, 0) = partptr_sub(nparts*(2*local_sub_ip-1) + ip, 1);
              }
              else if (ip != 0) {
                partptr_sub(sub_ip, 0) = partptr_sub(nparts*2*(n_subparts_per_part-1) + ip - 1, 1);
              }
              partptr_sub(sub_ip, 1) = last_sub_line_length + partptr_sub(sub_ip, 0);

              part2rowidx0_sub(sub_ip + 1) = part2rowidx0_sub(sub_ip) + last_sub_line_length;

#ifdef IFPACK2_BLOCKTRIDICONTAINER_USE_PRINTF
              printf("Sub Part index = %d, first LID associated to the sub part = %d, sub part size = %d;\n", sub_ip, partptr_sub(sub_ip, 0), last_sub_line_length);
#endif
            }
          }
        }

        {
          local_ordinal_type npacks = ceil(float(nparts)/vector_length);

          local_ordinal_type ip_max = nparts > vector_length ? vector_length : nparts;
          for (local_ordinal_type ip=0;ip<ip_max;++ip) {
            part2packrowidx0_sub(ip, 0) = 0;
          }
          for (local_ordinal_type ipack=0;ipack<npacks;++ipack) {
            if (ipack != 0) {
              local_ordinal_type ip_min = ipack*vector_length;
              ip_max = nparts > (ipack+1)*vector_length ? (ipack+1)*vector_length : nparts;
              for (local_ordinal_type ip=ip_min;ip<ip_max;++ip) {
                part2packrowidx0_sub(ip, 0) = part2packrowidx0_sub(ip-vector_length, part2packrowidx0_sub.extent(1)-1);
              }
            }

            for (size_type local_sub_ip=0; local_sub_ip<part2packrowidx0_sub.extent(1)-1;++local_sub_ip) {
              local_ordinal_type ip_min = ipack*vector_length;
              ip_max = nparts > (ipack+1)*vector_length ? (ipack+1)*vector_length : nparts;

              const local_ordinal_type full_line_length = partptr(ip_min+1) - partptr(ip_min);

              constexpr local_ordinal_type connection_length = 2;      

              const local_ordinal_type sub_line_length = (full_line_length - (n_subparts_per_part - 1) * connection_length) / n_subparts_per_part;
              const local_ordinal_type last_sub_line_length = full_line_length - (n_subparts_per_part - 1) * (connection_length + sub_line_length);

              if (local_sub_ip % 2 == 0) pack_nrows_sub = sub_line_length;
              if (local_sub_ip % 2 == 1) pack_nrows_sub = connection_length;
              if (local_sub_ip == part2packrowidx0_sub.extent(1)-2) pack_nrows_sub = last_sub_line_length;

              part2packrowidx0_sub(ip_min, local_sub_ip + 1) = part2packrowidx0_sub(ip_min, local_sub_ip) + pack_nrows_sub;

              for (local_ordinal_type ip=ip_min+1;ip<ip_max;++ip) {
                part2packrowidx0_sub(ip, local_sub_ip + 1) = part2packrowidx0_sub(ip_min, local_sub_ip + 1);
              }
            }
          }

          Kokkos::deep_copy(interf.part2packrowidx0_sub, part2packrowidx0_sub);
        }
        IFPACK2_BLOCKHELPER_TIMER_FENCE(typename BlockHelperDetails::ImplType<MatrixType>::execution_space)
      }
#if defined(BLOCKTRIDICONTAINER_DEBUG)
      TEUCHOS_ASSERT(partptr(nparts) == nrows);
#endif
      if (lclrow(0) != 0) interf.row_contiguous = false;

      Kokkos::deep_copy(interf.partptr, partptr);
      Kokkos::deep_copy(interf.lclrow, lclrow);

      Kokkos::deep_copy(interf.partptr_sub, partptr_sub);

      //assume No overlap. Thus:
      interf.part2rowidx0 = interf.partptr;
      Kokkos::deep_copy(interf.part2packrowidx0, part2packrowidx0);

      interf.part2packrowidx0_back = part2packrowidx0_sub(part2packrowidx0_sub.extent(0) - 1, part2packrowidx0_sub.extent(1) - 1);
      Kokkos::deep_copy(interf.rowidx2part, rowidx2part);

      { // Fill packptr.
        IFPACK2_BLOCKHELPER_TIMER("Fill packptr", packptr0);
        local_ordinal_type npacks = ceil(float(nparts)/vector_length) * (part2packrowidx0_sub.extent(1)-1);
        npacks = 0;
        for (local_ordinal_type ip=1;ip<=nparts;++ip) //n_sub_parts_and_schur
          if (part2packrowidx0(ip) != part2packrowidx0(ip-1))
            ++npacks;

        interf.packptr = local_ordinal_type_1d_view(do_not_initialize_tag("packptr"), npacks + 1);
        const auto packptr = Kokkos::create_mirror_view(interf.packptr);
        packptr(0) = 0;
        for (local_ordinal_type ip=1,k=1;ip<=nparts;++ip)
          if (part2packrowidx0(ip) != part2packrowidx0(ip-1))
            packptr(k++) = ip;
        
        Kokkos::deep_copy(interf.packptr, packptr);

        local_ordinal_type npacks_per_subpart = ceil(float(nparts)/vector_length);
        npacks = ceil(float(nparts)/vector_length) * (part2packrowidx0_sub.extent(1)-1);

        interf.packindices_sub = local_ordinal_type_1d_view(do_not_initialize_tag("packindices_sub"), npacks_per_subpart*n_subparts_per_part);
        interf.packindices_schur = local_ordinal_type_2d_view(do_not_initialize_tag("packindices_schur"), npacks_per_subpart,n_subparts_per_part-1);

        const auto packindices_sub = Kokkos::create_mirror_view(interf.packindices_sub);
        const auto packindices_schur = Kokkos::create_mirror_view(interf.packindices_schur);


        // Fill packindices_sub and packindices_schur
        for (local_ordinal_type local_sub_ip=0; local_sub_ip<n_subparts_per_part-1;++local_sub_ip) {
          for (local_ordinal_type local_pack_ip=0; local_pack_ip<npacks_per_subpart;++local_pack_ip) {
            packindices_sub(local_sub_ip * npacks_per_subpart + local_pack_ip) = 2 * local_sub_ip * npacks_per_subpart + local_pack_ip;
            packindices_schur(local_pack_ip,local_sub_ip) = 2 * local_sub_ip * npacks_per_subpart + local_pack_ip + npacks_per_subpart;
          }
        }

        for (local_ordinal_type local_pack_ip=0; local_pack_ip<npacks_per_subpart;++local_pack_ip) {
          packindices_sub((n_subparts_per_part-1) * npacks_per_subpart + local_pack_ip) = 2 * (n_subparts_per_part-1) * npacks_per_subpart + local_pack_ip;
        }

#ifdef IFPACK2_BLOCKTRIDICONTAINER_WRITE_MM
        std::cout << "packindices_sub = " << std::endl;
        for (size_type i = 0; i < packindices_sub.extent(0); ++i) {
            std::cout << packindices_sub(i) << " ";
        }
        std::cout << std::endl;
        std::cout << "packindices_sub end" << std::endl;

        std::cout << "packindices_schur = " << std::endl;
        for (size_type i = 0; i < packindices_schur.extent(0); ++i) {
          for (size_type j = 0; j < packindices_schur.extent(1); ++j) {
            std::cout << packindices_schur(i,j) << " ";
          }
          std::cout << std::endl;
        }
        
        std::cout << "packindices_schur end" << std::endl;
#endif

        Kokkos::deep_copy(interf.packindices_sub, packindices_sub);
        Kokkos::deep_copy(interf.packindices_schur, packindices_schur);

        interf.packptr_sub = local_ordinal_type_1d_view(do_not_initialize_tag("packptr"), npacks + 1);
        const auto packptr_sub = Kokkos::create_mirror_view(interf.packptr_sub);
        packptr_sub(0) = 0;
        for (local_ordinal_type k=0;k<npacks + 1;++k)
          packptr_sub(k) = packptr(k%npacks_per_subpart) + (k / npacks_per_subpart) * packptr(npacks_per_subpart);

        Kokkos::deep_copy(interf.packptr_sub, packptr_sub);
        IFPACK2_BLOCKHELPER_TIMER_FENCE(typename BlockHelperDetails::ImplType<MatrixType>::execution_space)
      }
      IFPACK2_BLOCKHELPER_TIMER_FENCE(typename BlockHelperDetails::ImplType<MatrixType>::execution_space)

      return interf;
    }

    ///
    /// block tridiagonals
    ///
    template <typename MatrixType>
    struct BlockTridiags {
      using impl_type = BlockHelperDetails::ImplType<MatrixType>;
      using local_ordinal_type_1d_view = typename impl_type::local_ordinal_type_1d_view;
      using size_type_1d_view = typename impl_type::size_type_1d_view;
      using size_type_2d_view = typename impl_type::size_type_2d_view;
      using vector_type_3d_view = typename impl_type::vector_type_3d_view;
      using vector_type_4d_view = typename impl_type::vector_type_4d_view;

      // flat_td_ptr(i) is the index into flat-array values of the start of the
      // i'th tridiag. pack_td_ptr is the same, but for packs. If vector_length ==
      // 1, pack_td_ptr is the same as flat_td_ptr; if vector_length > 1, then i %
      // vector_length is the position in the pack.
      size_type_2d_view flat_td_ptr, pack_td_ptr, pack_td_ptr_schur;
      // List of local column indices into A from which to grab
      // data. flat_td_ptr(i) points to the start of the i'th tridiag's data.
      local_ordinal_type_1d_view A_colindsub;
      // Tridiag block values. pack_td_ptr(i) points to the start of the i'th
      // tridiag's pack, and i % vector_length gives the position in the pack.
      vector_type_3d_view values;
      // Schur block values. pack_td_ptr_schur(i) points to the start of the i'th
      // Schur's pack, and i % vector_length gives the position in the pack.
      vector_type_3d_view values_schur;
      // inv(A_00)*A_01 block values.
      vector_type_4d_view e_values;

      bool is_diagonal_only;

      BlockTridiags() = default;
      BlockTridiags(const BlockTridiags &b) = default;

      // Index into row-major block of a tridiag.
      template <typename idx_type>
      static KOKKOS_FORCEINLINE_FUNCTION
      idx_type IndexToRow (const idx_type& ind) { return (ind + 1) / 3; }
      // Given a row of a row-major tridiag, return the index of the first block
      // in that row.
      template <typename idx_type>
      static KOKKOS_FORCEINLINE_FUNCTION
      idx_type RowToIndex (const idx_type& row) { return row > 0 ? 3*row - 1 : 0; }
      // Number of blocks in a tridiag having a given number of rows.
      template <typename idx_type>
      static KOKKOS_FORCEINLINE_FUNCTION
      idx_type NumBlocks (const idx_type& nrows) { return nrows > 0 ? 3*nrows - 2 : 0; }
      // Number of blocks associated to a Schur complement having a given number of rows.
      template <typename idx_type>
      static KOKKOS_FORCEINLINE_FUNCTION
      idx_type NumBlocksSchur (const idx_type& nrows) { return nrows > 0 ? 3*nrows + 2 : 0; }
    };


    ///
    /// block tridiags initialization from part interface
    ///
    template<typename MatrixType>
    BlockTridiags<MatrixType>
    createBlockTridiags(const BlockHelperDetails::PartInterface<MatrixType> &interf) {
      IFPACK2_BLOCKHELPER_TIMER("createBlockTridiags", createBlockTridiags0);
      using impl_type = BlockHelperDetails::ImplType<MatrixType>;
      using execution_space = typename impl_type::execution_space;
      using local_ordinal_type = typename impl_type::local_ordinal_type;
      using size_type = typename impl_type::size_type;
      using size_type_2d_view = typename impl_type::size_type_2d_view;

      constexpr int vector_length = impl_type::vector_length;

      BlockTridiags<MatrixType> btdm;

      const local_ordinal_type ntridiags = interf.partptr_sub.extent(0);

      { // construct the flat index pointers into the tridiag values array.
        btdm.flat_td_ptr = size_type_2d_view(do_not_initialize_tag("btdm.flat_td_ptr"), interf.nparts, 2*interf.n_subparts_per_part);
        const Kokkos::RangePolicy<execution_space> policy(0, 2 * interf.nparts * interf.n_subparts_per_part );
        Kokkos::parallel_scan
          ("createBlockTridiags::RangePolicy::flat_td_ptr",
           policy, KOKKOS_LAMBDA(const local_ordinal_type &i, size_type &update, const bool &final) {
            const local_ordinal_type partidx = i/(2 * interf.n_subparts_per_part);
            const local_ordinal_type local_subpartidx = i % (2 * interf.n_subparts_per_part);

            if (final) {
              btdm.flat_td_ptr(partidx, local_subpartidx) = update;
            }
            if (local_subpartidx != (2 * interf.n_subparts_per_part -1)) {
              const local_ordinal_type nrows = interf.partptr_sub(interf.nparts*local_subpartidx + partidx,1) - interf.partptr_sub(interf.nparts*local_subpartidx + partidx,0);
              if (local_subpartidx % 2 == 0)
                update += btdm.NumBlocks(nrows);
              else
                update += btdm.NumBlocksSchur(nrows);
            }
          });

        const auto nblocks = Kokkos::create_mirror_view_and_copy
          (Kokkos::HostSpace(), Kokkos::subview(btdm.flat_td_ptr, interf.nparts-1, 2*interf.n_subparts_per_part-1));
        btdm.is_diagonal_only = (static_cast<local_ordinal_type>(nblocks()) == ntridiags);
      }

      // And the packed index pointers.
      if (vector_length == 1) {
        btdm.pack_td_ptr = btdm.flat_td_ptr;
      } else {
        //const local_ordinal_type npacks = interf.packptr_sub.extent(0) - 1;

        local_ordinal_type npacks_per_subpart = 0;
        const auto part2packrowidx0 = Kokkos::create_mirror_view(interf.part2packrowidx0);
        Kokkos::deep_copy(part2packrowidx0, interf.part2packrowidx0);
        for (local_ordinal_type ip=1;ip<=interf.nparts;++ip) //n_sub_parts_and_schur
            if (part2packrowidx0(ip) != part2packrowidx0(ip-1))
              ++npacks_per_subpart;

        btdm.pack_td_ptr = size_type_2d_view(do_not_initialize_tag("btdm.pack_td_ptr"), interf.nparts, 2*interf.n_subparts_per_part);
        const Kokkos::RangePolicy<execution_space> policy(0,npacks_per_subpart);

        Kokkos::parallel_for
          ("createBlockTridiags::RangePolicy::pack_td_ptr",
           policy, KOKKOS_LAMBDA(const local_ordinal_type &i) {
            for (local_ordinal_type j = 0; j < 2*interf.n_subparts_per_part; ++j) {
              const local_ordinal_type pack_id = ( j == 2*interf.n_subparts_per_part-1 ) ? i+(j-1)*npacks_per_subpart : i+j*npacks_per_subpart;
              const local_ordinal_type nparts_in_pack = interf.packptr_sub(pack_id+1) - interf.packptr_sub(pack_id);

              const local_ordinal_type parti = interf.packptr_sub(pack_id);
              const local_ordinal_type partidx = parti%interf.nparts;

              for (local_ordinal_type pti=0;pti<nparts_in_pack;++pti) {
                btdm.pack_td_ptr(partidx+pti, j) = btdm.flat_td_ptr(i, j);
              }
            }
          });
      }

      btdm.pack_td_ptr_schur = size_type_2d_view(do_not_initialize_tag("btdm.pack_td_ptr_schur"), interf.nparts, interf.n_subparts_per_part);

      const auto host_pack_td_ptr_schur = Kokkos::create_mirror_view(btdm.pack_td_ptr_schur);
      constexpr local_ordinal_type connection_length = 2;

      host_pack_td_ptr_schur(0,0) = 0;
      for (local_ordinal_type i = 0; i < interf.nparts; ++i) {
        if (i % vector_length == 0) {
          if (i != 0)
            host_pack_td_ptr_schur(i,0) = host_pack_td_ptr_schur(i-1,host_pack_td_ptr_schur.extent(1)-1);
          for (local_ordinal_type j = 0; j < interf.n_subparts_per_part-1; ++j) {
            host_pack_td_ptr_schur(i,j+1) = host_pack_td_ptr_schur(i,j) + btdm.NumBlocks(connection_length) + (j != 0 ? 1 : 0) + (j != interf.n_subparts_per_part-2 ? 1 : 0);
          }
        }
        else {
          for (local_ordinal_type j = 0; j < interf.n_subparts_per_part; ++j) {
            host_pack_td_ptr_schur(i,j) = host_pack_td_ptr_schur(i-1,j);
          }
        }
      }

      Kokkos::deep_copy(btdm.pack_td_ptr_schur, host_pack_td_ptr_schur);

#ifdef IFPACK2_BLOCKTRIDICONTAINER_WRITE_MM
      const auto host_flat_td_ptr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), btdm.flat_td_ptr);
      std::cout << "flat_td_ptr = " << std::endl;
      for (size_type i = 0; i < host_flat_td_ptr.extent(0); ++i) {
        for (size_type j = 0; j < host_flat_td_ptr.extent(1); ++j) {
          std::cout << host_flat_td_ptr(i,j) << " ";
        }
        std::cout << std::endl;
      }
      std::cout << "flat_td_ptr end" << std::endl;

      const auto host_pack_td_ptr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), btdm.pack_td_ptr);

      std::cout << "pack_td_ptr = " << std::endl;
      for (size_type i = 0; i < host_pack_td_ptr.extent(0); ++i) {
        for (size_type j = 0; j < host_pack_td_ptr.extent(1); ++j) {
          std::cout << host_pack_td_ptr(i,j) << " ";
        }
        std::cout << std::endl;
      }
      std::cout << "pack_td_ptr end" << std::endl;


      std::cout << "pack_td_ptr_schur = " << std::endl;
      for (size_type i = 0; i < host_pack_td_ptr_schur.extent(0); ++i) {
        for (size_type j = 0; j < host_pack_td_ptr_schur.extent(1); ++j) {
          std::cout << host_pack_td_ptr_schur(i,j) << " ";
        }
        std::cout << std::endl;
      }
      std::cout << "pack_td_ptr_schur end" << std::endl;
#endif

      // values and A_colindsub are created in the symbolic phase
      IFPACK2_BLOCKHELPER_TIMER_FENCE(typename BlockHelperDetails::ImplType<MatrixType>::execution_space)

      return btdm;
    }

    // Set the tridiags to be I to the full pack block size. That way, if a
    // tridiag within a pack is shorter than the longest one, the extra blocks are
    // processed in a safe way. Similarly, in the solve phase, if the extra blocks
    // in the packed multvector are 0, and the tridiag LU reflects the extra I
    // blocks, then the solve proceeds as though the extra blocks aren't
    // present. Since this extra work is part of the SIMD calls, it's not actually
    // extra work. Instead, it means we don't have to put checks or masks in, or
    // quiet NaNs. This functor has to be called just once, in the symbolic phase,
    // since the numeric phase fills in only the used entries, leaving these I
    // blocks intact.
    template<typename MatrixType>
    void
    setTridiagsToIdentity
      (const BlockTridiags<MatrixType>& btdm,
       const typename BlockHelperDetails::ImplType<MatrixType>::local_ordinal_type_1d_view& packptr)
    {
      using impl_type = BlockHelperDetails::ImplType<MatrixType>;
      using execution_space = typename impl_type::execution_space;
      using local_ordinal_type = typename impl_type::local_ordinal_type;
      using size_type_2d_view = typename impl_type::size_type_2d_view;

      const ConstUnmanaged<size_type_2d_view> pack_td_ptr(btdm.pack_td_ptr);
      const local_ordinal_type blocksize = btdm.values.extent(1);

      {
        const int vector_length = impl_type::vector_length;
        const int internal_vector_length = impl_type::internal_vector_length;

        using btdm_scalar_type = typename impl_type::btdm_scalar_type;
        using internal_vector_type = typename impl_type::internal_vector_type;
        using internal_vector_type_4d_view =
          typename impl_type::internal_vector_type_4d_view;

        using team_policy_type = Kokkos::TeamPolicy<execution_space>;
        const internal_vector_type_4d_view values
          (reinterpret_cast<internal_vector_type*>(btdm.values.data()),
           btdm.values.extent(0),
           btdm.values.extent(1),
           btdm.values.extent(2),
           vector_length/internal_vector_length);
        const local_ordinal_type vector_loop_size = values.extent(3);
#if defined(KOKKOS_ENABLE_CUDA) && defined(__CUDA_ARCH__)
        local_ordinal_type total_team_size(0);
        if      (blocksize <=  5) total_team_size =  32;
        else if (blocksize <=  9) total_team_size =  64;
        else if (blocksize <= 12) total_team_size =  96;
        else if (blocksize <= 16) total_team_size = 128;
        else if (blocksize <= 20) total_team_size = 160;
        else                      total_team_size = 160;
        const local_ordinal_type team_size = total_team_size/vector_loop_size;
        const team_policy_type policy(packptr.extent(0)-1, team_size, vector_loop_size);
#elif defined(KOKKOS_ENABLE_HIP)
	// FIXME: HIP
	// These settings might be completely wrong
	// will have to do some experiments to decide
	// what makes sense on AMD GPUs
        local_ordinal_type total_team_size(0);
        if      (blocksize <=  5) total_team_size =  32;
        else if (blocksize <=  9) total_team_size =  64;
        else if (blocksize <= 12) total_team_size =  96;
        else if (blocksize <= 16) total_team_size = 128;
        else if (blocksize <= 20) total_team_size = 160;
        else                      total_team_size = 160;
        const local_ordinal_type team_size = total_team_size/vector_loop_size;
        const team_policy_type policy(packptr.extent(0)-1, team_size, vector_loop_size);
#elif defined(KOKKOS_ENABLE_SYCL)
	// SYCL: FIXME
	        local_ordinal_type total_team_size(0);
        if      (blocksize <=  5) total_team_size =  32;
        else if (blocksize <=  9) total_team_size =  64;
        else if (blocksize <= 12) total_team_size =  96;
        else if (blocksize <= 16) total_team_size = 128;
        else if (blocksize <= 20) total_team_size = 160;
        else                      total_team_size = 160;
        const local_ordinal_type team_size = total_team_size/vector_loop_size;
        const team_policy_type policy(packptr.extent(0)-1, team_size, vector_loop_size);
#else
	// Host architecture: team size is always one
        const team_policy_type policy(packptr.extent(0)-1, 1, 1);
#endif
        Kokkos::parallel_for
          ("setTridiagsToIdentity::TeamPolicy",
           policy, KOKKOS_LAMBDA(const typename team_policy_type::member_type &member) {
            const local_ordinal_type k = member.league_rank();
            const local_ordinal_type ibeg = pack_td_ptr(packptr(k),0);
            const local_ordinal_type iend = pack_td_ptr(packptr(k),pack_td_ptr.extent(1)-1);

            const local_ordinal_type diff = iend - ibeg;
            const local_ordinal_type icount = diff/3 + (diff%3 > 0);
            const btdm_scalar_type one(1);
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, vector_loop_size),[&](const int &v) {
                Kokkos::parallel_for(Kokkos::TeamThreadRange(member,icount),[&](const local_ordinal_type &ii) {
                    const local_ordinal_type i = ibeg + ii*3;
                    for (local_ordinal_type j=0;j<blocksize;++j) {
                      values(i,j,j,v) = one;
                    }
                  });
              });
          });
      }
    }

    ///
    /// symbolic phase, on host : create R = A - D, pack D
    ///
    template<typename MatrixType>
    void
    performSymbolicPhase(const Teuchos::RCP<const typename BlockHelperDetails::ImplType<MatrixType>::tpetra_row_matrix_type> &A,
                         const Teuchos::RCP<const typename BlockHelperDetails::ImplType<MatrixType>::tpetra_crs_graph_type> &g,
                         const BlockHelperDetails::PartInterface<MatrixType> &interf,
                         BlockTridiags<MatrixType> &btdm) {
      IFPACK2_BLOCKHELPER_TIMER("BlockTriDi::SymbolicPhase", SymbolicPhase);

      using impl_type = BlockHelperDetails::ImplType<MatrixType>;

      // using node_memory_space = typename impl_type::node_memory_space;
      using host_execution_space = typename impl_type::host_execution_space;

      using local_ordinal_type = typename impl_type::local_ordinal_type;
      using global_ordinal_type = typename impl_type::global_ordinal_type;
      using size_type = typename impl_type::size_type;
      using local_ordinal_type_1d_view = typename impl_type::local_ordinal_type_1d_view;
      using vector_type_3d_view = typename impl_type::vector_type_3d_view;
      using vector_type_4d_view = typename impl_type::vector_type_4d_view;
      using crs_matrix_type = typename impl_type::tpetra_crs_matrix_type;
      using block_crs_matrix_type = typename impl_type::tpetra_block_crs_matrix_type;

      constexpr int vector_length = impl_type::vector_length;

      const auto comm = A->getRowMap()->getComm();

      auto A_crs = Teuchos::rcp_dynamic_cast<const crs_matrix_type>(A);
      auto A_bcrs = Teuchos::rcp_dynamic_cast<const block_crs_matrix_type>(A);

      bool hasBlockCrsMatrix = ! A_bcrs.is_null ();
      TEUCHOS_ASSERT(hasBlockCrsMatrix || g->getLocalNumRows() != 0);
      const local_ordinal_type blocksize = hasBlockCrsMatrix ? A->getBlockSize() : A->getLocalNumRows()/g->getLocalNumRows();
      

      // mirroring to host
      const auto partptr = Kokkos::create_mirror_view_and_copy     (Kokkos::HostSpace(), interf.partptr);
      const auto lclrow = Kokkos::create_mirror_view_and_copy      (Kokkos::HostSpace(), interf.lclrow);
      const auto rowidx2part = Kokkos::create_mirror_view_and_copy (Kokkos::HostSpace(), interf.rowidx2part);
      const auto part2rowidx0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), interf.part2rowidx0);
      const auto packptr = Kokkos::create_mirror_view_and_copy     (Kokkos::HostSpace(), interf.packptr);

      const local_ordinal_type nrows = partptr(partptr.extent(0) - 1);

      Kokkos::View<local_ordinal_type*,host_execution_space> col2row("col2row", A->getLocalNumCols());

      // find column to row map on host
      
      Kokkos::deep_copy(col2row, Teuchos::OrdinalTraits<local_ordinal_type>::invalid());
      {
        const auto rowmap = g->getRowMap();
        const auto colmap = g->getColMap();
        const auto dommap = g->getDomainMap();
        TEUCHOS_ASSERT( !(rowmap.is_null() || colmap.is_null() || dommap.is_null()));

#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__) && !defined(__SYCL_DEVICE_ONLY__)
        const Kokkos::RangePolicy<host_execution_space> policy(0,nrows);
        Kokkos::parallel_for
          ("performSymbolicPhase::RangePolicy::col2row",
           policy, KOKKOS_LAMBDA(const local_ordinal_type &lr) {
            const global_ordinal_type gid = rowmap->getGlobalElement(lr);
            TEUCHOS_ASSERT(gid != Teuchos::OrdinalTraits<global_ordinal_type>::invalid());
            if (dommap->isNodeGlobalElement(gid)) {
              const local_ordinal_type lc = colmap->getLocalElement(gid);
#  if defined(BLOCKTRIDICONTAINER_DEBUG)
              TEUCHOS_TEST_FOR_EXCEPT_MSG(lc == Teuchos::OrdinalTraits<local_ordinal_type>::invalid(),
                                          BlockHelperDetails::get_msg_prefix(comm) << "GID " << gid
                                          << " gives an invalid local column.");
#  endif
              col2row(lc) = lr;
            }
          });
#endif
      }

      // construct the D and R graphs in A = D + R.
      {
        const auto local_graph = g->getLocalGraphHost();
        const auto local_graph_rowptr = local_graph.row_map;
        TEUCHOS_ASSERT(local_graph_rowptr.size() == static_cast<size_t>(nrows + 1));
        const auto local_graph_colidx = local_graph.entries;

        //assume no overlap.

        Kokkos::View<local_ordinal_type*,host_execution_space> lclrow2idx("lclrow2idx", nrows);
        {
          const Kokkos::RangePolicy<host_execution_space> policy(0,nrows);
          Kokkos::parallel_for
            ("performSymbolicPhase::RangePolicy::lclrow2idx",
             policy, KOKKOS_LAMBDA(const local_ordinal_type &i) {
              lclrow2idx[lclrow(i)] = i;
            });
        }

        // count (block) nnzs in D and R.
        typedef BlockHelperDetails::SumReducer<size_type,3,host_execution_space> sum_reducer_type;
        typename sum_reducer_type::value_type sum_reducer_value;
        {
          const Kokkos::RangePolicy<host_execution_space> policy(0,nrows);
          Kokkos::parallel_reduce
            // profiling interface does not work
            (//"performSymbolicPhase::RangePolicy::count_nnz",
             policy, KOKKOS_LAMBDA(const local_ordinal_type &lr, typename sum_reducer_type::value_type &update) {
              // LID -> index.
              const local_ordinal_type ri0 = lclrow2idx[lr];
              const local_ordinal_type pi0 = rowidx2part(ri0);
              for (size_type j=local_graph_rowptr(lr);j<local_graph_rowptr(lr+1);++j) {
                const local_ordinal_type lc = local_graph_colidx(j);
                const local_ordinal_type lc2r = col2row[lc];
                bool incr_R = false;
                do { // breakable
                  if (lc2r == (local_ordinal_type) -1) {
                    incr_R = true;
                    break;
                  }
                  const local_ordinal_type ri = lclrow2idx[lc2r];
                  const local_ordinal_type pi = rowidx2part(ri);
                  if (pi != pi0) {
                    incr_R = true;
                    break;
                  }
                  // Test for being in the tridiag. This is done in index space. In
                  // LID space, tridiag LIDs in a row are not necessarily related by
                  // {-1, 0, 1}.
                  if (ri0 + 1 >= ri && ri0 <= ri + 1)
                    ++update.v[0]; // D_nnz
                  else
                    incr_R = true;
                } while (0);
                if (incr_R) {
                  if (lc < nrows) ++update.v[1]; // R_nnz_owned
                  else ++update.v[2]; // R_nnz_remote
                }
              }
            }, sum_reducer_type(sum_reducer_value));
        }
        size_type D_nnz = sum_reducer_value.v[0];

        // construct the D_00 graph.
        {
          const auto flat_td_ptr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), btdm.flat_td_ptr);

          btdm.A_colindsub = local_ordinal_type_1d_view("btdm.A_colindsub", D_nnz);
          const auto D_A_colindsub = Kokkos::create_mirror_view(btdm.A_colindsub);

#if defined(BLOCKTRIDICONTAINER_DEBUG)
          Kokkos::deep_copy(D_A_colindsub, Teuchos::OrdinalTraits<local_ordinal_type>::invalid());
#endif

          const local_ordinal_type nparts = partptr.extent(0) - 1;

          {
            const Kokkos::RangePolicy<host_execution_space> policy(0, nparts);
            Kokkos::parallel_for
              ("performSymbolicPhase::RangePolicy<host_execution_space>::D_graph",
               policy, KOKKOS_LAMBDA(const local_ordinal_type &pi0) {
                const local_ordinal_type part_ri0 = part2rowidx0(pi0);
                local_ordinal_type offset = 0;
                for (local_ordinal_type ri0=partptr(pi0);ri0<partptr(pi0+1);++ri0) {
                  const local_ordinal_type td_row_os = btdm.RowToIndex(ri0 - part_ri0) + offset;
                  offset = 1;
                  const local_ordinal_type lr0 = lclrow(ri0);
                  const size_type j0 = local_graph_rowptr(lr0);
                  for (size_type j=j0;j<local_graph_rowptr(lr0+1);++j) {
                    const local_ordinal_type lc = local_graph_colidx(j);
                    const local_ordinal_type lc2r = col2row[lc];
                    if (lc2r == (local_ordinal_type) -1) continue;
                    const local_ordinal_type ri = lclrow2idx[lc2r];
                    const local_ordinal_type pi = rowidx2part(ri);
                    if (pi != pi0) continue;
                    if (ri + 1 < ri0 || ri > ri0 + 1) continue;
                    const local_ordinal_type row_entry = j - j0;
                    D_A_colindsub(flat_td_ptr(pi0,0) + ((td_row_os + ri) - ri0)) = row_entry;
                  }
                }
              });
          }
#if defined(BLOCKTRIDICONTAINER_DEBUG)
          for (size_t i=0;i<D_A_colindsub.extent(0);++i)
            TEUCHOS_ASSERT(D_A_colindsub(i) != Teuchos::OrdinalTraits<local_ordinal_type>::invalid());
#endif
          Kokkos::deep_copy(btdm.A_colindsub, D_A_colindsub);

          // Allocate values.
          {
            const auto pack_td_ptr_last = Kokkos::subview(btdm.pack_td_ptr, btdm.pack_td_ptr.extent(0)-1, btdm.pack_td_ptr.extent(1)-1);
            const auto num_packed_blocks = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), pack_td_ptr_last);
            btdm.values = vector_type_3d_view("btdm.values", num_packed_blocks(), blocksize, blocksize);

            if (interf.n_subparts_per_part > 1) {
              const auto pack_td_ptr_schur_last = Kokkos::subview(btdm.pack_td_ptr_schur, btdm.pack_td_ptr_schur.extent(0)-1, btdm.pack_td_ptr_schur.extent(1)-1);
              const auto num_packed_blocks_schur = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), pack_td_ptr_schur_last);
              btdm.values_schur = vector_type_3d_view("btdm.values_schur", num_packed_blocks_schur(), blocksize, blocksize);
            }

            if (vector_length > 1) setTridiagsToIdentity(btdm, interf.packptr);
          }
        }

        // Allocate view for E and initialize the values with B:
        
        if (interf.n_subparts_per_part > 1)
          btdm.e_values = vector_type_4d_view("btdm.e_values", 2, interf.part2packrowidx0_back, blocksize, blocksize);
      }
      IFPACK2_BLOCKHELPER_TIMER_FENCE(typename BlockHelperDetails::ImplType<MatrixType>::execution_space)
    }


    ///
    /// numeric phase, initialize the preconditioner
    ///
    template<typename ArgActiveExecutionMemorySpace>
    struct ExtractAndFactorizeTridiagsDefaultModeAndAlgo;

    template<>
    struct ExtractAndFactorizeTridiagsDefaultModeAndAlgo<Kokkos::HostSpace> {
      typedef KB::Mode::Serial mode_type;
#if defined(__KOKKOSBATCHED_INTEL_MKL_COMPACT_BATCHED__)
      typedef KB::Algo::Level3::CompactMKL algo_type;
#else
      typedef KB::Algo::Level3::Blocked algo_type;
#endif
      static int recommended_team_size(const int /* blksize */,
                                       const int /* vector_length */,
                                       const int /* internal_vector_length */) {
        return 1;
      }

    };

#if defined(KOKKOS_ENABLE_CUDA)
    static inline int ExtractAndFactorizeRecommendedCudaTeamSize(const int blksize,
                                                                 const int vector_length,
                                                                 const int internal_vector_length) {
      const int vector_size = vector_length/internal_vector_length;
      int total_team_size(0);
      if      (blksize <=  5) total_team_size =  32;
      else if (blksize <=  9) total_team_size =  32; // 64
      else if (blksize <= 12) total_team_size =  96;
      else if (blksize <= 16) total_team_size = 128;
      else if (blksize <= 20) total_team_size = 160;
      else                    total_team_size = 160;
      return 2*total_team_size/vector_size;
    }
    template<>
    struct ExtractAndFactorizeTridiagsDefaultModeAndAlgo<Kokkos::CudaSpace> {
      typedef KB::Mode::Team mode_type;
      typedef KB::Algo::Level3::Unblocked algo_type;
      static int recommended_team_size(const int blksize,
                                       const int vector_length,
                                       const int internal_vector_length) {
        return ExtractAndFactorizeRecommendedCudaTeamSize(blksize, vector_length, internal_vector_length);
      }
    };
    template<>
    struct ExtractAndFactorizeTridiagsDefaultModeAndAlgo<Kokkos::CudaUVMSpace> {
      typedef KB::Mode::Team mode_type;
      typedef KB::Algo::Level3::Unblocked algo_type;
      static int recommended_team_size(const int blksize,
                                       const int vector_length,
                                       const int internal_vector_length) {
        return ExtractAndFactorizeRecommendedCudaTeamSize(blksize, vector_length, internal_vector_length);
      }
    };
#endif

#if defined(KOKKOS_ENABLE_HIP)
    static inline int ExtractAndFactorizeRecommendedHIPTeamSize(const int blksize,
								const int vector_length,
								const int internal_vector_length) {
      const int vector_size = vector_length/internal_vector_length;
      int total_team_size(0);
      if      (blksize <=  5) total_team_size =  32;
      else if (blksize <=  9) total_team_size =  32; // 64
      else if (blksize <= 12) total_team_size =  96;
      else if (blksize <= 16) total_team_size = 128;
      else if (blksize <= 20) total_team_size = 160;
      else                    total_team_size = 160;
      return 2*total_team_size/vector_size;
    }
    template<>
    struct ExtractAndFactorizeTridiagsDefaultModeAndAlgo<Kokkos::HIPSpace> {
      typedef KB::Mode::Team mode_type;
      typedef KB::Algo::Level3::Unblocked algo_type;
      static int recommended_team_size(const int blksize,
                                       const int vector_length,
                                       const int internal_vector_length) {
        return ExtractAndFactorizeRecommendedHIPTeamSize(blksize, vector_length, internal_vector_length);
      }
    };
    template<>
    struct ExtractAndFactorizeTridiagsDefaultModeAndAlgo<Kokkos::HIPHostPinnedSpace> {
      typedef KB::Mode::Team mode_type;
      typedef KB::Algo::Level3::Unblocked algo_type;
      static int recommended_team_size(const int blksize,
                                       const int vector_length,
                                       const int internal_vector_length) {
        return ExtractAndFactorizeRecommendedHIPTeamSize(blksize, vector_length, internal_vector_length);
      }
    };
#endif

#if defined(KOKKOS_ENABLE_SYCL)
    static inline int ExtractAndFactorizeRecommendedSYCLTeamSize(const int blksize,
								const int vector_length,
								const int internal_vector_length) {
      const int vector_size = vector_length/internal_vector_length;
      int total_team_size(0);
      if      (blksize <=  5) total_team_size =  32;
      else if (blksize <=  9) total_team_size =  32; // 64
      else if (blksize <= 12) total_team_size =  96;
      else if (blksize <= 16) total_team_size = 128;
      else if (blksize <= 20) total_team_size = 160;
      else                    total_team_size = 160;
      return 2*total_team_size/vector_size;
    }
    template<>
    struct ExtractAndFactorizeTridiagsDefaultModeAndAlgo<Kokkos::Experimental::SYCLDeviceUSMSpace> {
      typedef KB::Mode::Team mode_type;
      typedef KB::Algo::Level3::Unblocked algo_type;
      static int recommended_team_size(const int blksize,
                                       const int vector_length,
                                       const int internal_vector_length) {
        return ExtractAndFactorizeRecommendedSYCLTeamSize(blksize, vector_length, internal_vector_length);
      }
    };
    template<>
    struct ExtractAndFactorizeTridiagsDefaultModeAndAlgo<Kokkos::Experimental::SYCLSharedUSMSpace> {
      typedef KB::Mode::Team mode_type;
      typedef KB::Algo::Level3::Unblocked algo_type;
      static int recommended_team_size(const int blksize,
                                       const int vector_length,
                                       const int internal_vector_length) {
        return ExtractAndFactorizeRecommendedSYCLTeamSize(blksize, vector_length, internal_vector_length);
      }
    };
#endif

    template<typename impl_type, typename WWViewType>
    KOKKOS_INLINE_FUNCTION
    void
    solveMultiVector(const typename Kokkos::TeamPolicy<typename impl_type::execution_space>::member_type &member,
                      const typename impl_type::local_ordinal_type &/* blocksize */,
                      const typename impl_type::local_ordinal_type &i0,
                      const typename impl_type::local_ordinal_type &r0,
                      const typename impl_type::local_ordinal_type &nrows,
                      const typename impl_type::local_ordinal_type &v,
                      const ConstUnmanaged<typename impl_type::internal_vector_type_4d_view> D_internal_vector_values,
                      const Unmanaged<typename impl_type::internal_vector_type_4d_view> X_internal_vector_values,
                      const WWViewType &WW, 
                      const bool skip_first_pass=false) {
        using execution_space = typename impl_type::execution_space;
        using team_policy_type = Kokkos::TeamPolicy<execution_space>;
        using member_type = typename team_policy_type::member_type;
        using local_ordinal_type = typename impl_type::local_ordinal_type;

        typedef SolveTridiagsDefaultModeAndAlgo
          <typename execution_space::memory_space> default_mode_and_algo_type;

        typedef typename default_mode_and_algo_type::mode_type default_mode_type;
        typedef typename default_mode_and_algo_type::multi_vector_algo_type default_algo_type;

        using btdm_magnitude_type = typename impl_type::btdm_magnitude_type;

        // constant
        const auto one = Kokkos::ArithTraits<btdm_magnitude_type>::one();
        const auto zero = Kokkos::ArithTraits<btdm_magnitude_type>::zero();

        // subview pattern
        auto A  = Kokkos::subview(D_internal_vector_values, i0, Kokkos::ALL(), Kokkos::ALL(), v);
        auto X1 = Kokkos::subview(X_internal_vector_values, r0, Kokkos::ALL(), Kokkos::ALL(), v);
        auto X2 = X1;

        local_ordinal_type i = i0, r = r0;


        if (nrows > 1) {
          // solve Lx = x
          if (skip_first_pass) {
            i += (nrows-2) * 3;
            r += (nrows-2);
            A.assign_data( &D_internal_vector_values(i+2,0,0,v) );
            X2.assign_data( &X_internal_vector_values(++r,0,0,v) );
            A.assign_data( &D_internal_vector_values(i+3,0,0,v) );
            KB::Trsm<member_type,
                    KB::Side::Left,KB::Uplo::Lower,KB::Trans::NoTranspose,KB::Diag::Unit,
                    default_mode_type,default_algo_type>
              ::invoke(member, one, A, X2);
            X1.assign_data( X2.data() );
            i+=3;
          }
          else {
            KB::Trsm<member_type,
                    KB::Side::Left,KB::Uplo::Lower,KB::Trans::NoTranspose,KB::Diag::Unit,
                    default_mode_type,default_algo_type>
              ::invoke(member, one, A, X1);
            for (local_ordinal_type tr=1;tr<nrows;++tr,i+=3) {
              A.assign_data( &D_internal_vector_values(i+2,0,0,v) );
              X2.assign_data( &X_internal_vector_values(++r,0,0,v) );
              member.team_barrier();
              KB::Gemm<member_type,
                      KB::Trans::NoTranspose,KB::Trans::NoTranspose,
                      default_mode_type,default_algo_type>
                ::invoke(member, -one, A, X1, one, X2);
              A.assign_data( &D_internal_vector_values(i+3,0,0,v) );
              KB::Trsm<member_type,
                      KB::Side::Left,KB::Uplo::Lower,KB::Trans::NoTranspose,KB::Diag::Unit,
                      default_mode_type,default_algo_type>
                ::invoke(member, one, A, X2);
              X1.assign_data( X2.data() );
            }
          }

          // solve Ux = x
          KB::Trsm<member_type,
                   KB::Side::Left,KB::Uplo::Upper,KB::Trans::NoTranspose,KB::Diag::NonUnit,
                   default_mode_type,default_algo_type>
            ::invoke(member, one, A, X1);
          for (local_ordinal_type tr=nrows;tr>1;--tr) {
            i -= 3;
            A.assign_data( &D_internal_vector_values(i+1,0,0,v) );
            X2.assign_data( &X_internal_vector_values(--r,0,0,v) );
            member.team_barrier();
            KB::Gemm<member_type,
                     KB::Trans::NoTranspose,KB::Trans::NoTranspose,
                     default_mode_type,default_algo_type>
              ::invoke(member, -one, A, X1, one, X2);

            A.assign_data( &D_internal_vector_values(i,0,0,v) );
            KB::Trsm<member_type,
                     KB::Side::Left,KB::Uplo::Upper,KB::Trans::NoTranspose,KB::Diag::NonUnit,
                     default_mode_type,default_algo_type>
              ::invoke(member, one, A, X2);
            X1.assign_data( X2.data() );
          }
        } else {
          // matrix is already inverted
          auto W = Kokkos::subview(WW, Kokkos::ALL(), Kokkos::ALL(), v);
          KB::Copy<member_type,KB::Trans::NoTranspose,default_mode_type>
            ::invoke(member, X1, W);
          member.team_barrier();
          KB::Gemm<member_type,
                   KB::Trans::NoTranspose,KB::Trans::NoTranspose,
                   default_mode_type,default_algo_type>
            ::invoke(member, one, A, W, zero, X1);
        }

    }

    template<typename impl_type, typename WWViewType, typename XViewType>
    KOKKOS_INLINE_FUNCTION
    void
    solveSingleVectorNew(const typename Kokkos::TeamPolicy<typename impl_type::execution_space>::member_type &member,
                      const typename impl_type::local_ordinal_type &blocksize,
                      const typename impl_type::local_ordinal_type &i0,
                      const typename impl_type::local_ordinal_type &r0,
                      const typename impl_type::local_ordinal_type &nrows,
                      const typename impl_type::local_ordinal_type &v,
                      const ConstUnmanaged<typename impl_type::internal_vector_type_4d_view> D_internal_vector_values,
                      const XViewType &X_internal_vector_values, //Unmanaged<typename impl_type::internal_vector_type_4d_view>
                      const WWViewType &WW) {
      using execution_space = typename impl_type::execution_space;
      //using team_policy_type = Kokkos::TeamPolicy<execution_space>;
      //using member_type = typename team_policy_type::member_type;
      using local_ordinal_type = typename impl_type::local_ordinal_type;

      typedef SolveTridiagsDefaultModeAndAlgo
        <typename execution_space::memory_space> default_mode_and_algo_type;

      typedef typename default_mode_and_algo_type::mode_type default_mode_type;
      typedef typename default_mode_and_algo_type::single_vector_algo_type default_algo_type;

      using btdm_magnitude_type = typename impl_type::btdm_magnitude_type;

      // base pointers
      auto A = D_internal_vector_values.data();
      auto X = X_internal_vector_values.data();

      // constant
      const auto one = Kokkos::ArithTraits<btdm_magnitude_type>::one();
      const auto zero = Kokkos::ArithTraits<btdm_magnitude_type>::zero();
      //const local_ordinal_type num_vectors = X_scalar_values.extent(2);

      // const local_ordinal_type blocksize = D_scalar_values.extent(1);
      const local_ordinal_type astep = D_internal_vector_values.stride_0();
      const local_ordinal_type as0 = D_internal_vector_values.stride_1(); //blocksize*vector_length;
      const local_ordinal_type as1 = D_internal_vector_values.stride_2(); //vector_length;
      const local_ordinal_type xstep = X_internal_vector_values.stride_0();
      const local_ordinal_type xs0 = X_internal_vector_values.stride_1(); //vector_length;

      // move to starting point
      A += i0*astep + v;
      X += r0*xstep + v;

      //for (local_ordinal_type col=0;col<num_vectors;++col)
      if (nrows > 1) {
        // solve Lx = x
        KOKKOSBATCHED_TRSV_LOWER_NO_TRANSPOSE_INTERNAL_INVOKE
          (default_mode_type,default_algo_type,
            member,
            KB::Diag::Unit,
            blocksize,blocksize,
            one,
            A, as0, as1,
            X, xs0);

        for (local_ordinal_type tr=1;tr<nrows;++tr) {
          member.team_barrier();
          KOKKOSBATCHED_GEMV_NO_TRANSPOSE_INTERNAL_INVOKE
            (default_mode_type,default_algo_type,
              member,
              blocksize, blocksize,
              -one,
              A+2*astep, as0, as1,
              X, xs0,
              one,
              X+1*xstep, xs0);
          KOKKOSBATCHED_TRSV_LOWER_NO_TRANSPOSE_INTERNAL_INVOKE
            (default_mode_type,default_algo_type,
              member,
              KB::Diag::Unit,
              blocksize,blocksize,
              one,
              A+3*astep, as0, as1,
              X+1*xstep, xs0);

          A += 3*astep;
          X += 1*xstep;
        }

        // solve Ux = x
        KOKKOSBATCHED_TRSV_UPPER_NO_TRANSPOSE_INTERNAL_INVOKE
          (default_mode_type,default_algo_type,
            member,
            KB::Diag::NonUnit,
            blocksize, blocksize,
            one,
            A, as0, as1,
            X, xs0);

        for (local_ordinal_type tr=nrows;tr>1;--tr) {
          A -= 3*astep;
          member.team_barrier();
          KOKKOSBATCHED_GEMV_NO_TRANSPOSE_INTERNAL_INVOKE
            (default_mode_type,default_algo_type,
              member,
              blocksize, blocksize,
              -one,
              A+1*astep, as0, as1,
              X, xs0,
              one,
              X-1*xstep, xs0);
          KOKKOSBATCHED_TRSV_UPPER_NO_TRANSPOSE_INTERNAL_INVOKE
            (default_mode_type,default_algo_type,
              member,
              KB::Diag::NonUnit,
              blocksize, blocksize,
              one,
              A, as0, as1,
              X-1*xstep,xs0);
          X -= 1*xstep;
        }
        // for multiple rhs
        //X += xs1;
      } else {
        const local_ordinal_type ws0 = WW.stride_0();
        auto W = WW.data() + v;
        KOKKOSBATCHED_COPY_VECTOR_NO_TRANSPOSE_INTERNAL_INVOKE
          (default_mode_type,
            member, blocksize, X, xs0, W, ws0);
        member.team_barrier();
        KOKKOSBATCHED_GEMV_NO_TRANSPOSE_INTERNAL_INVOKE
          (default_mode_type,default_algo_type,
            member,
            blocksize, blocksize,
            one,
            A, as0, as1,
            W, xs0,
            zero,
            X, xs0);
      }
    }

    template<typename local_ordinal_type, typename ViewType>
    void writeBTDValuesToFile (const local_ordinal_type &n_parts, const ViewType &scalar_values_device, std::string fileName) {
#ifdef IFPACK2_BLOCKTRIDICONTAINER_WRITE_MM
      auto scalar_values = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), scalar_values_device);
      std::ofstream myfile;
      myfile.open (fileName);

      const local_ordinal_type n_parts_per_pack = n_parts < (local_ordinal_type) scalar_values.extent(3) ? n_parts : scalar_values.extent(3);
      local_ordinal_type nnz = scalar_values.extent(0) * scalar_values.extent(1) * scalar_values.extent(2) * n_parts_per_pack;
      const local_ordinal_type n_blocks = scalar_values.extent(0)*n_parts_per_pack;
      const local_ordinal_type n_blocks_per_part = n_blocks/n_parts;

      const local_ordinal_type block_size = scalar_values.extent(1);

      const local_ordinal_type n_rows_per_part = (n_blocks_per_part+2)/3 * block_size;
      const local_ordinal_type n_rows = n_rows_per_part*n_parts;

      const local_ordinal_type n_packs = ceil(float(n_parts)/n_parts_per_pack);

      myfile << "%%MatrixMarket matrix coordinate real general"<< std::endl;
      myfile << "%%nnz = " << nnz; 
      myfile << " block size = " << block_size;
      myfile << " number of blocks = " << n_blocks;
      myfile << " number of parts = " << n_parts;
      myfile << " number of blocks per part = " << n_blocks_per_part;
      myfile << " number of rows = " << n_rows ;
      myfile << " number of cols = " << n_rows;
      myfile << " number of packs = " << n_packs << std::endl;

      myfile << n_rows << " " << n_rows << " " << nnz << std::setprecision(9) << std::endl;

      local_ordinal_type current_part_idx, current_block_idx, current_row_offset, current_col_offset, current_row, current_col;
      for (local_ordinal_type i_pack=0;i_pack<n_packs;++i_pack) {
        for (local_ordinal_type i_part_in_pack=0;i_part_in_pack<n_parts_per_pack;++i_part_in_pack) {
          current_part_idx = i_part_in_pack + i_pack * n_parts_per_pack;
          for (local_ordinal_type i_block_in_part=0;i_block_in_part<n_blocks_per_part;++i_block_in_part) {
            current_block_idx = i_block_in_part + i_pack * n_blocks_per_part;
            if (current_block_idx >= (local_ordinal_type) scalar_values.extent(0))
              continue;
            if (i_block_in_part % 3 == 0) {
              current_row_offset = i_block_in_part/3 * block_size;
              current_col_offset = i_block_in_part/3 * block_size;
            }
            else if (i_block_in_part % 3 == 1) {
              current_row_offset = (i_block_in_part-1)/3 * block_size;
              current_col_offset = ((i_block_in_part-1)/3+1) * block_size;
            }
            else if (i_block_in_part % 3 == 2) {
              current_row_offset = ((i_block_in_part-2)/3+1) * block_size;
              current_col_offset = (i_block_in_part-2)/3 * block_size;
            }
            current_row_offset += current_part_idx * n_rows_per_part;
            current_col_offset += current_part_idx * n_rows_per_part;
            for (local_ordinal_type i_in_block=0;i_in_block<block_size;++i_in_block) {
              for (local_ordinal_type j_in_block=0;j_in_block<block_size;++j_in_block) {
                current_row = current_row_offset + i_in_block + 1;
                current_col = current_col_offset + j_in_block + 1;
                myfile <<  current_row << " " << current_col << " " << scalar_values(current_block_idx,i_in_block,j_in_block,i_part_in_pack) << std::endl;
              }
            }
          }
        }
      }

      myfile.close();
#endif
    }

    template<typename local_ordinal_type, typename ViewType>
    void write4DMultiVectorValuesToFile (const local_ordinal_type &n_parts, const ViewType &scalar_values_device, std::string fileName) {
#ifdef IFPACK2_BLOCKTRIDICONTAINER_WRITE_MM
      auto scalar_values = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), scalar_values_device);
      std::ofstream myfile;
      myfile.open (fileName);

      const local_ordinal_type n_parts_per_pack = n_parts < scalar_values.extent(3) ? n_parts : scalar_values.extent(3);
      const local_ordinal_type n_blocks = scalar_values.extent(0)*n_parts_per_pack;
      const local_ordinal_type n_blocks_per_part = n_blocks/n_parts;

      const local_ordinal_type block_size = scalar_values.extent(1);
      const local_ordinal_type n_cols = scalar_values.extent(2);

      const local_ordinal_type n_rows_per_part = n_blocks_per_part * block_size;
      const local_ordinal_type n_rows = n_rows_per_part*n_parts;

      const local_ordinal_type n_packs = ceil(float(n_parts)/n_parts_per_pack);


      myfile << "%%MatrixMarket matrix array real general"<< std::endl;
      myfile << "%%block size = " << block_size;
      myfile << " number of blocks = " << n_blocks;
      myfile << " number of parts = " << n_parts;
      myfile << " number of blocks per part = " << n_blocks_per_part;
      myfile << " number of rows = " << n_rows ;
      myfile << " number of cols = " << n_cols;
      myfile << " number of packs = " << n_packs << std::endl;

      myfile << n_rows << " " << n_cols << std::setprecision(9) << std::endl;     

      local_ordinal_type current_part_idx, current_block_idx, current_row_offset;
      (void) current_row_offset;
      (void) current_part_idx;
      for (local_ordinal_type j_in_block=0;j_in_block<n_cols;++j_in_block) {      
        for (local_ordinal_type i_pack=0;i_pack<n_packs;++i_pack) {
          for (local_ordinal_type i_part_in_pack=0;i_part_in_pack<n_parts_per_pack;++i_part_in_pack) {
            current_part_idx = i_part_in_pack + i_pack * n_parts_per_pack;
            for (local_ordinal_type i_block_in_part=0;i_block_in_part<n_blocks_per_part;++i_block_in_part) {
              current_block_idx = i_block_in_part + i_pack * n_blocks_per_part;

              if (current_block_idx >= (local_ordinal_type) scalar_values.extent(0))
                continue;
              for (local_ordinal_type i_in_block=0;i_in_block<block_size;++i_in_block) {
                myfile << scalar_values(current_block_idx,i_in_block,j_in_block,i_part_in_pack) << std::endl;
              }
            }
          }
        }
      }
      myfile.close();
#endif
    }

    template<typename local_ordinal_type, typename ViewType>
    void write5DMultiVectorValuesToFile (const local_ordinal_type &n_parts, const ViewType &scalar_values_device, std::string fileName) {
#ifdef IFPACK2_BLOCKTRIDICONTAINER_WRITE_MM
      auto scalar_values = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), scalar_values_device);
      std::ofstream myfile;
      myfile.open (fileName);

      const local_ordinal_type n_parts_per_pack = n_parts < scalar_values.extent(4) ? n_parts : scalar_values.extent(4);
      const local_ordinal_type n_blocks = scalar_values.extent(1)*n_parts_per_pack;
      const local_ordinal_type n_blocks_per_part = n_blocks/n_parts;

      const local_ordinal_type block_size = scalar_values.extent(2);
      const local_ordinal_type n_blocks_cols = scalar_values.extent(0);
      const local_ordinal_type n_cols = n_blocks_cols * block_size;

      const local_ordinal_type n_rows_per_part = n_blocks_per_part * block_size;
      const local_ordinal_type n_rows = n_rows_per_part*n_parts;

      const local_ordinal_type n_packs = ceil(float(n_parts)/n_parts_per_pack);

      myfile << "%%MatrixMarket matrix array real general"<< std::endl;
      myfile << "%%block size = " << block_size;
      myfile << " number of blocks = " << n_blocks;
      myfile << " number of parts = " << n_parts;
      myfile << " number of blocks per part = " << n_blocks_per_part;
      myfile << " number of rows = " << n_rows ;
      myfile << " number of cols = " << n_cols;
      myfile << " number of packs = " << n_packs << std::endl;

      myfile << n_rows << " " << n_cols << std::setprecision(9) << std::endl;     

      local_ordinal_type current_part_idx, current_block_idx, current_row_offset;
      (void) current_row_offset;
      (void) current_part_idx;
      for (local_ordinal_type i_block_col=0;i_block_col<n_blocks_cols;++i_block_col) {
        for (local_ordinal_type j_in_block=0;j_in_block<block_size;++j_in_block) {      
          for (local_ordinal_type i_pack=0;i_pack<n_packs;++i_pack) {
            for (local_ordinal_type i_part_in_pack=0;i_part_in_pack<n_parts_per_pack;++i_part_in_pack) {
              current_part_idx = i_part_in_pack + i_pack * n_parts_per_pack;
              for (local_ordinal_type i_block_in_part=0;i_block_in_part<n_blocks_per_part;++i_block_in_part) {
                current_block_idx = i_block_in_part + i_pack * n_blocks_per_part;

                if (current_block_idx >= (local_ordinal_type) scalar_values.extent(1))
                  continue;
                for (local_ordinal_type i_in_block=0;i_in_block<block_size;++i_in_block) {
                  myfile << scalar_values(i_block_col,current_block_idx,i_in_block,j_in_block,i_part_in_pack) << std::endl;
                }
              }
            }
          }
        }
      }
      myfile.close();
#endif
    }

    template<typename local_ordinal_type, typename member_type, typename ViewType1, typename ViewType2>
    KOKKOS_INLINE_FUNCTION
    void
    copy3DView(const member_type &member, const ViewType1 &view1, const ViewType2 &view2) {
/*
      // Kokkos::Experimental::local_deep_copy
      auto teamVectorRange =
          Kokkos::TeamVectorMDRange<Kokkos::Rank<3>, member_type>(
              member, view1.extent(0), view1.extent(1), view1.extent(2));

      Kokkos::parallel_for
        (teamVectorRange,
      [&](const local_ordinal_type &i, const local_ordinal_type &j, const local_ordinal_type &k) {
        view1(i,j,k) = view2(i,j,k);
      });
*/
      Kokkos::Experimental::local_deep_copy(member, view1, view2);
    }
    template<typename MatrixType>
    struct ExtractAndFactorizeTridiags {
    public:
      using impl_type = BlockHelperDetails::ImplType<MatrixType>;
      // a functor cannot have both device_type and execution_space; specialization error in kokkos
      using execution_space = typename impl_type::execution_space;
      using memory_space = typename impl_type::memory_space;
      /// scalar types
      using local_ordinal_type = typename impl_type::local_ordinal_type;
      using size_type = typename impl_type::size_type;
      using impl_scalar_type = typename impl_type::impl_scalar_type;
      using magnitude_type = typename impl_type::magnitude_type;
      /// tpetra interface
      using row_matrix_type = typename impl_type::tpetra_row_matrix_type;
      using crs_graph_type = typename impl_type::tpetra_crs_graph_type;
      /// views
      using local_ordinal_type_1d_view = typename impl_type::local_ordinal_type_1d_view;
      using local_ordinal_type_2d_view = typename impl_type::local_ordinal_type_2d_view;
      using size_type_2d_view = typename impl_type::size_type_2d_view;
      using impl_scalar_type_1d_view_tpetra = typename impl_type::impl_scalar_type_1d_view_tpetra;
      /// vectorization
      using btdm_scalar_type = typename impl_type::btdm_scalar_type;
      using btdm_magnitude_type = typename impl_type::btdm_magnitude_type;
      using vector_type_3d_view = typename impl_type::vector_type_3d_view;
      using vector_type_4d_view = typename impl_type::vector_type_4d_view;
      using internal_vector_type_4d_view = typename impl_type::internal_vector_type_4d_view;
      using internal_vector_type_5d_view = typename impl_type::internal_vector_type_5d_view;
      using btdm_scalar_type_4d_view = typename impl_type::btdm_scalar_type_4d_view;
      using btdm_scalar_type_5d_view = typename impl_type::btdm_scalar_type_5d_view;
      using internal_vector_scratch_type_3d_view = Scratch<typename impl_type::internal_vector_type_3d_view>;
      using btdm_scalar_scratch_type_3d_view = Scratch<typename impl_type::btdm_scalar_type_3d_view>;

      using internal_vector_type = typename impl_type::internal_vector_type;
      static constexpr int vector_length = impl_type::vector_length;
      static constexpr int internal_vector_length = impl_type::internal_vector_length;

      /// team policy member type
      using team_policy_type = Kokkos::TeamPolicy<execution_space>;
      using member_type = typename team_policy_type::member_type;

    private:
      // part interface
      const ConstUnmanaged<local_ordinal_type_1d_view> partptr, lclrow, packptr, packindices_sub, packptr_sub;
      const ConstUnmanaged<local_ordinal_type_2d_view> partptr_sub, part2packrowidx0_sub, packindices_schur;
      const local_ordinal_type max_partsz;
      // block crs matrix (it could be Kokkos::UVMSpace::size_type, which is int)
      using size_type_1d_view_tpetra = Kokkos::View<size_t*,typename impl_type::node_device_type>;
      ConstUnmanaged<size_type_1d_view_tpetra> A_block_rowptr;
      ConstUnmanaged<size_type_1d_view_tpetra> A_point_rowptr;
      ConstUnmanaged<impl_scalar_type_1d_view_tpetra> A_values;
      // block tridiags
      const ConstUnmanaged<size_type_2d_view> pack_td_ptr, flat_td_ptr, pack_td_ptr_schur;
      const ConstUnmanaged<local_ordinal_type_1d_view> A_colindsub;
      const Unmanaged<internal_vector_type_4d_view> internal_vector_values, internal_vector_values_schur;
      const Unmanaged<internal_vector_type_5d_view> e_internal_vector_values;
      const Unmanaged<btdm_scalar_type_4d_view> scalar_values, scalar_values_schur;
      const Unmanaged<btdm_scalar_type_5d_view> e_scalar_values;
      // shared information
      const local_ordinal_type blocksize, blocksize_square;
      // diagonal safety
      const magnitude_type tiny;
      const local_ordinal_type vector_loop_size;
      const local_ordinal_type vector_length_value;

      bool hasBlockCrsMatrix;

    public:
      ExtractAndFactorizeTridiags(const BlockTridiags<MatrixType> &btdm_,
                                  const BlockHelperDetails::PartInterface<MatrixType> &interf_,
                                  const Teuchos::RCP<const row_matrix_type> &A_,
                                  const Teuchos::RCP<const crs_graph_type> &G_,
                                  const magnitude_type& tiny_) :
        // interface
        partptr(interf_.partptr),
        lclrow(interf_.lclrow),
        packptr(interf_.packptr),
        packindices_sub(interf_.packindices_sub),
        packptr_sub(interf_.packptr_sub),
        partptr_sub(interf_.partptr_sub),
        part2packrowidx0_sub(interf_.part2packrowidx0_sub),
        packindices_schur(interf_.packindices_schur),
        max_partsz(interf_.max_partsz),
        // block tridiags
        pack_td_ptr(btdm_.pack_td_ptr),
        flat_td_ptr(btdm_.flat_td_ptr),
        pack_td_ptr_schur(btdm_.pack_td_ptr_schur),
        A_colindsub(btdm_.A_colindsub),
        internal_vector_values((internal_vector_type*)btdm_.values.data(),
                               btdm_.values.extent(0),
                               btdm_.values.extent(1),
                               btdm_.values.extent(2),
                               vector_length/internal_vector_length),
        internal_vector_values_schur((internal_vector_type*)btdm_.values_schur.data(),
                               btdm_.values_schur.extent(0),
                               btdm_.values_schur.extent(1),
                               btdm_.values_schur.extent(2),
                               vector_length/internal_vector_length),
        e_internal_vector_values((internal_vector_type*)btdm_.e_values.data(),
                                btdm_.e_values.extent(0),
                                btdm_.e_values.extent(1),
                                btdm_.e_values.extent(2),
                                btdm_.e_values.extent(3),
                                vector_length/internal_vector_length),
        scalar_values((btdm_scalar_type*)btdm_.values.data(),
                      btdm_.values.extent(0),
                      btdm_.values.extent(1),
                      btdm_.values.extent(2),
                      vector_length),
        scalar_values_schur((btdm_scalar_type*)btdm_.values_schur.data(),
                      btdm_.values_schur.extent(0),
                      btdm_.values_schur.extent(1),
                      btdm_.values_schur.extent(2),
                      vector_length),
        e_scalar_values((btdm_scalar_type*)btdm_.e_values.data(),
                      btdm_.e_values.extent(0),
                      btdm_.e_values.extent(1),
                      btdm_.e_values.extent(2),
                      btdm_.e_values.extent(3),
                      vector_length),
        blocksize(btdm_.values.extent(1)),
        blocksize_square(blocksize*blocksize),
        // diagonal weight to avoid zero pivots
        tiny(tiny_),
        vector_loop_size(vector_length/internal_vector_length),
        vector_length_value(vector_length) {
          using crs_matrix_type = typename impl_type::tpetra_crs_matrix_type;
          using block_crs_matrix_type = typename impl_type::tpetra_block_crs_matrix_type;

          auto A_crs = Teuchos::rcp_dynamic_cast<const crs_matrix_type>(A_);
          auto A_bcrs = Teuchos::rcp_dynamic_cast<const block_crs_matrix_type>(A_);

          hasBlockCrsMatrix = ! A_bcrs.is_null ();

          A_block_rowptr = G_->getLocalGraphDevice().row_map;
          if (hasBlockCrsMatrix) {
            A_values = const_cast<block_crs_matrix_type*>(A_bcrs.get())->getValuesDeviceNonConst();
          }
          else {
            A_point_rowptr = A_crs->getCrsGraph()->getLocalGraphDevice().row_map;
            A_values = A_crs->getLocalValuesDevice (Tpetra::Access::ReadOnly);
          }
        }

    private:

      KOKKOS_INLINE_FUNCTION
      void
      extract(local_ordinal_type partidx,
              local_ordinal_type local_subpartidx,
              local_ordinal_type npacks) const {
#ifdef IFPACK2_BLOCKTRIDICONTAINER_USE_PRINTF
        printf("extract partidx = %d, local_subpartidx = %d, npacks = %d;\n", partidx, local_subpartidx, npacks);
#endif
        using tlb = BlockHelperDetails::TpetraLittleBlock<Tpetra::Impl::BlockCrsMatrixLittleBlockArrayLayout>;
        const size_type kps = pack_td_ptr(partidx, local_subpartidx);
        local_ordinal_type kfs[vector_length] = {};
        local_ordinal_type ri0[vector_length] = {};
        local_ordinal_type nrows[vector_length] = {};

        for (local_ordinal_type vi=0;vi<npacks;++vi,++partidx) {
          kfs[vi] = flat_td_ptr(partidx,local_subpartidx);
          ri0[vi] = partptr_sub(pack_td_ptr.extent(0)*local_subpartidx + partidx,0);
          nrows[vi] = partptr_sub(pack_td_ptr.extent(0)*local_subpartidx + partidx,1) - ri0[vi];
#ifdef IFPACK2_BLOCKTRIDICONTAINER_USE_PRINTF
          printf("kfs[%d] = %d;\n", vi, kfs[vi]);
          printf("ri0[%d] = %d;\n", vi, ri0[vi]);
          printf("nrows[%d] = %d;\n", vi, nrows[vi]);
#endif
        }
        local_ordinal_type tr_min = 0;
        local_ordinal_type tr_max = nrows[0];
        if (local_subpartidx % 2 == 1) {
          tr_min -= 1;
          tr_max += 1;
        }
#ifdef IFPACK2_BLOCKTRIDICONTAINER_USE_PRINTF
        printf("tr_min = %d and tr_max = %d;\n", tr_min, tr_max);
#endif
        for (local_ordinal_type tr=tr_min,j=0;tr<tr_max;++tr) {
          for (local_ordinal_type e=0;e<3;++e) {
            if (hasBlockCrsMatrix) {
              const impl_scalar_type* block[vector_length] = {};
              for (local_ordinal_type vi=0;vi<npacks;++vi) {
                const size_type Aj = A_block_rowptr(lclrow(ri0[vi] + tr)) + A_colindsub(kfs[vi] + j);

                block[vi] = &A_values(Aj*blocksize_square);
              }
              const size_type pi = kps + j;
#ifdef IFPACK2_BLOCKTRIDICONTAINER_USE_PRINTF
              printf("Extract pi = %ld, ri0 + tr = %d, kfs + j = %d\n", pi, ri0[0] + tr, kfs[0] + j);
#endif            
              ++j;            
              for (local_ordinal_type ii=0;ii<blocksize;++ii) {
                for (local_ordinal_type jj=0;jj<blocksize;++jj) {
                  const auto idx = tlb::getFlatIndex(ii, jj, blocksize);
                  auto& v = internal_vector_values(pi, ii, jj, 0);
                  for (local_ordinal_type vi=0;vi<npacks;++vi) {
                    v[vi] = static_cast<btdm_scalar_type>(block[vi][idx]);
                  }
                }
              }
            }
            else {
              const size_type pi = kps + j;

              for (local_ordinal_type vi=0;vi<npacks;++vi) {
                const size_type Aj_c = A_colindsub(kfs[vi] + j);

                for (local_ordinal_type ii=0;ii<blocksize;++ii) {
                  auto point_row_offset = A_point_rowptr(lclrow(ri0[vi] + tr)*blocksize + ii);

                  for (local_ordinal_type jj=0;jj<blocksize;++jj) {
                    scalar_values(pi, ii, jj, vi) = A_values(point_row_offset + Aj_c*blocksize + jj);
                  }
                }
              }
              ++j;
            }
            if (nrows[0] == 1) break;
            if (local_subpartidx % 2 == 0) {
              if (e == 1 && (tr == 0 || tr+1 == nrows[0])) break;
              for (local_ordinal_type vi=1;vi<npacks;++vi) {
                if ((e == 0 && nrows[vi] == 1) || (e == 1 && tr+1 == nrows[vi])) {
                  npacks = vi;
                  break;
                }
              }
            }
            else {
              if (e == 0 && (tr == -1 || tr == nrows[0])) break;
              for (local_ordinal_type vi=1;vi<npacks;++vi) {
                if ((e == 0 && nrows[vi] == 1) || (e == 0 && tr == nrows[vi])) {
                  npacks = vi;
                  break;
                }
              }
            }
          }
        }
      }

      KOKKOS_INLINE_FUNCTION
      void
      extract(const member_type &member,
              const local_ordinal_type &partidxbeg,
              local_ordinal_type local_subpartidx,
              const local_ordinal_type &npacks,
              const local_ordinal_type &vbeg) const {
#ifdef IFPACK2_BLOCKTRIDICONTAINER_USE_PRINTF
        printf("extract partidxbeg = %d, local_subpartidx = %d, npacks = %d, vbeg = %d;\n", partidxbeg, local_subpartidx, npacks, vbeg);                
#endif
        using tlb = BlockHelperDetails::TpetraLittleBlock<Tpetra::Impl::BlockCrsMatrixLittleBlockArrayLayout>;
        local_ordinal_type kfs_vals[internal_vector_length] = {};
        local_ordinal_type ri0_vals[internal_vector_length] = {};
        local_ordinal_type nrows_vals[internal_vector_length] = {};

        const size_type kps = pack_td_ptr(partidxbeg,local_subpartidx);
        for (local_ordinal_type v=vbeg,vi=0;v<npacks && vi<internal_vector_length;++v,++vi) {
          kfs_vals[vi] = flat_td_ptr(partidxbeg+vi,local_subpartidx);
          ri0_vals[vi] = partptr_sub(pack_td_ptr.extent(0)*local_subpartidx + partidxbeg+vi,0);
          nrows_vals[vi] = partptr_sub(pack_td_ptr.extent(0)*local_subpartidx + partidxbeg+vi,1) - ri0_vals[vi];
#ifdef IFPACK2_BLOCKTRIDICONTAINER_USE_PRINTF
          printf("kfs_vals[%d] = %d;\n", vi, kfs_vals[vi]);
          printf("ri0_vals[%d] = %d;\n", vi, ri0_vals[vi]);
          printf("nrows_vals[%d] = %d;\n", vi, nrows_vals[vi]);
#endif
        }

        local_ordinal_type j_vals[internal_vector_length] = {};

        local_ordinal_type tr_min = 0;
        local_ordinal_type tr_max = nrows_vals[0];
        if (local_subpartidx % 2 == 1) {
          tr_min -= 1;
          tr_max += 1;
        }
#ifdef IFPACK2_BLOCKTRIDICONTAINER_USE_PRINTF
        printf("tr_min = %d and tr_max = %d;\n", tr_min, tr_max);
#endif
        for (local_ordinal_type tr=tr_min;tr<tr_max;++tr) {
          for (local_ordinal_type v=vbeg,vi=0;v<npacks && vi<internal_vector_length;++v,++vi) {
            const local_ordinal_type nrows = (local_subpartidx % 2 == 0 ? nrows_vals[vi] : nrows_vals[vi]);
            if ((local_subpartidx % 2 == 0 && tr < nrows) || (local_subpartidx % 2 == 1 && tr < nrows+1)) {
              auto &j = j_vals[vi];
              const local_ordinal_type kfs = kfs_vals[vi];
              const local_ordinal_type ri0 = ri0_vals[vi];
              local_ordinal_type lbeg, lend;
              if (local_subpartidx % 2 == 0) {
                lbeg = (tr == tr_min    ? 1 : 0);
                lend = (tr == nrows - 1 ? 2 : 3);
              }
              else {
                lbeg = 0;
                lend = 3;
                if (tr == tr_min) {
                  lbeg = 1;
                  lend = 2;
                }
                else if (tr == nrows) {
                  lbeg = 0;
                  lend = 1;
                }
              }
              if (hasBlockCrsMatrix) {
                for (local_ordinal_type l=lbeg;l<lend;++l,++j) {
                  const size_type Aj = A_block_rowptr(lclrow(ri0 + tr)) + A_colindsub(kfs + j);
                  const impl_scalar_type* block = &A_values(Aj*blocksize_square);
                  const size_type pi = kps + j;
#ifdef IFPACK2_BLOCKTRIDICONTAINER_USE_PRINTF
                  printf("Extract pi = %ld, ri0 + tr = %d, kfs + j = %d, tr = %d, lbeg = %d, lend = %d, l = %d\n", pi, ri0 + tr, kfs + j, tr, lbeg, lend, l);
#endif
                  Kokkos::parallel_for
                    (Kokkos::TeamThreadRange(member,blocksize),
                    [&](const local_ordinal_type &ii) {
                      for (local_ordinal_type jj=0;jj<blocksize;++jj) {
                        scalar_values(pi, ii, jj, v) = static_cast<btdm_scalar_type>(block[tlb::getFlatIndex(ii,jj,blocksize)]);
                      }
                    });
                }
              }
              else {
                for (local_ordinal_type l=lbeg;l<lend;++l,++j) {
                  const size_type Aj_c = A_colindsub(kfs + j);
                  const size_type pi = kps + j;
                  Kokkos::parallel_for
                    (Kokkos::TeamThreadRange(member,blocksize),
                    [&](const local_ordinal_type &ii) {
                      auto point_row_offset = A_point_rowptr(lclrow(ri0 + tr)*blocksize + ii);
                      for (local_ordinal_type jj=0;jj<blocksize;++jj) {
                        scalar_values(pi, ii, jj, v) = A_values(point_row_offset + Aj_c*blocksize + jj);
                      }
                    });
                }
              }
            }
          }
        }
      }

      template<typename AAViewType,
               typename WWViewType>
      KOKKOS_INLINE_FUNCTION
      void
      factorize_subline(const member_type &member,
                const local_ordinal_type &i0,
                const local_ordinal_type &nrows,
                const local_ordinal_type &v,
                const AAViewType &AA,
                const WWViewType &WW) const {

        typedef ExtractAndFactorizeTridiagsDefaultModeAndAlgo
          <typename execution_space::memory_space> default_mode_and_algo_type;

        typedef typename default_mode_and_algo_type::mode_type default_mode_type;
        typedef typename default_mode_and_algo_type::algo_type default_algo_type;

        // constant
        const auto one = Kokkos::ArithTraits<btdm_magnitude_type>::one();

#ifdef IFPACK2_BLOCKTRIDICONTAINER_USE_PRINTF
        printf("i0 = %d, nrows = %d, v = %d, AA.extent(0) = %ld;\n", i0, nrows, v, AA.extent(0));
#endif

        // subview pattern
        auto A = Kokkos::subview(AA, i0, Kokkos::ALL(), Kokkos::ALL(), v);
        KB::LU<member_type,
               default_mode_type,KB::Algo::LU::Unblocked>
          ::invoke(member, A , tiny);

        if (nrows > 1) {
          auto B = A;
          auto C = A;
          local_ordinal_type i = i0;
          for (local_ordinal_type tr=1;tr<nrows;++tr,i+=3) {
#ifdef IFPACK2_BLOCKTRIDICONTAINER_USE_PRINTF
            printf("tr = %d, i = %d;\n", tr, i);
#endif
            B.assign_data( &AA(i+1,0,0,v) );
            KB::Trsm<member_type,
                     KB::Side::Left,KB::Uplo::Lower,KB::Trans::NoTranspose,KB::Diag::Unit,
                     default_mode_type,default_algo_type>
              ::invoke(member, one, A, B);
            C.assign_data( &AA(i+2,0,0,v) );
            KB::Trsm<member_type,
                     KB::Side::Right,KB::Uplo::Upper,KB::Trans::NoTranspose,KB::Diag::NonUnit,
                     default_mode_type,default_algo_type>
              ::invoke(member, one, A, C);
            A.assign_data( &AA(i+3,0,0,v) );

            member.team_barrier();
            KB::Gemm<member_type,
                     KB::Trans::NoTranspose,KB::Trans::NoTranspose,
                     default_mode_type,default_algo_type>
              ::invoke(member, -one, C, B, one, A);
            KB::LU<member_type,
                   default_mode_type,KB::Algo::LU::Unblocked>
              ::invoke(member, A, tiny);
          }
        } else {
          // for block jacobi invert a matrix here
          auto W = Kokkos::subview(WW, Kokkos::ALL(), Kokkos::ALL(), v);
          KB::Copy<member_type,KB::Trans::NoTranspose,default_mode_type>
            ::invoke(member, A, W);
          KB::SetIdentity<member_type,default_mode_type>
            ::invoke(member, A);
          member.team_barrier();
          KB::Trsm<member_type,
                   KB::Side::Left,KB::Uplo::Lower,KB::Trans::NoTranspose,KB::Diag::Unit,
                   default_mode_type,default_algo_type>
            ::invoke(member, one, W, A);
          KB::Trsm<member_type,
                   KB::Side::Left,KB::Uplo::Upper,KB::Trans::NoTranspose,KB::Diag::NonUnit,
                   default_mode_type,default_algo_type>
            ::invoke(member, one, W, A);
        }
      }

    public:

      struct ExtractAndFactorizeSubLineTag {};
      struct ExtractBCDTag {};
      struct ComputeETag {};
      struct ComputeSchurTag {};
      struct FactorizeSchurTag {};

      KOKKOS_INLINE_FUNCTION
      void
      operator() (const ExtractAndFactorizeSubLineTag &, const member_type &member) const {
        // btdm is packed and sorted from largest one
        const local_ordinal_type packidx = packindices_sub(member.league_rank());

        const local_ordinal_type subpartidx = packptr_sub(packidx);
        const local_ordinal_type n_parts = part2packrowidx0_sub.extent(0);
        const local_ordinal_type local_subpartidx = subpartidx/n_parts;
        const local_ordinal_type partidx = subpartidx%n_parts;

        const local_ordinal_type npacks = packptr_sub(packidx+1) - subpartidx;
        const local_ordinal_type i0 = pack_td_ptr(partidx,local_subpartidx);
        const local_ordinal_type nrows = partptr_sub(subpartidx,1) - partptr_sub(subpartidx,0);

        internal_vector_scratch_type_3d_view
          WW(member.team_scratch(0), blocksize, blocksize, vector_loop_size);

#ifdef IFPACK2_BLOCKTRIDICONTAINER_USE_PRINTF
        printf("rank = %d, i0 = %d, npacks = %d, nrows = %d, packidx = %d, subpartidx = %d, partidx = %d, local_subpartidx = %d;\n", member.league_rank(), i0, npacks, nrows, packidx, subpartidx, partidx, local_subpartidx);
        printf("vector_loop_size = %d\n", vector_loop_size);
#endif

        if (vector_loop_size == 1) {
          extract(partidx, local_subpartidx, npacks);
          factorize_subline(member, i0, nrows, 0, internal_vector_values, WW);
        } else {
          Kokkos::parallel_for
            (Kokkos::ThreadVectorRange(member, vector_loop_size),
	     [&](const local_ordinal_type &v) {
              const local_ordinal_type vbeg = v*internal_vector_length;
#ifdef IFPACK2_BLOCKTRIDICONTAINER_USE_PRINTF
              printf("i0 = %d, npacks = %d, vbeg = %d;\n", i0, npacks, vbeg);
#endif
              if (vbeg < npacks)
                extract(member, partidx+vbeg, local_subpartidx, npacks, vbeg);
              // this is not safe if vector loop size is different from vector size of 
              // the team policy. we always make sure this when constructing the team policy
              member.team_barrier();
              factorize_subline(member, i0, nrows, v, internal_vector_values, WW);
            });
        }
      }

      KOKKOS_INLINE_FUNCTION
      void
      operator() (const ExtractBCDTag &, const member_type &member) const {
        // btdm is packed and sorted from largest one
        const local_ordinal_type packindices_schur_i = member.league_rank() % packindices_schur.extent(0);
        const local_ordinal_type packindices_schur_j = member.league_rank() / packindices_schur.extent(0);
        const local_ordinal_type packidx = packindices_schur(packindices_schur_i, packindices_schur_j);

        const local_ordinal_type subpartidx = packptr_sub(packidx);
        const local_ordinal_type n_parts = part2packrowidx0_sub.extent(0);
        const local_ordinal_type local_subpartidx = subpartidx/n_parts;
        const local_ordinal_type partidx = subpartidx%n_parts;

        const local_ordinal_type npacks = packptr_sub(packidx+1) - subpartidx;
        //const local_ordinal_type i0 = pack_td_ptr(partidx,local_subpartidx);
        //const local_ordinal_type nrows = partptr_sub(subpartidx,1) - partptr_sub(subpartidx,0);

        if (vector_loop_size == 1) {
          extract(partidx, local_subpartidx, npacks);
        }
        else {
          Kokkos::parallel_for
            (Kokkos::ThreadVectorRange(member, vector_loop_size),
	     [&](const local_ordinal_type &v) {
              const local_ordinal_type vbeg = v*internal_vector_length;
#ifdef IFPACK2_BLOCKTRIDICONTAINER_USE_PRINTF
              const local_ordinal_type i0 = pack_td_ptr(partidx,local_subpartidx);
              printf("i0 = %d, npacks = %d, vbeg = %d;\n", i0, npacks, vbeg);
#endif
              if (vbeg < npacks)
                extract(member, partidx+vbeg, local_subpartidx, npacks, vbeg);
            });
        }

        member.team_barrier();

        const size_type kps1 = pack_td_ptr(partidx, local_subpartidx);
        const size_type kps2 = pack_td_ptr(partidx, local_subpartidx+1)-1;

        const local_ordinal_type r1 = part2packrowidx0_sub(partidx,local_subpartidx)-1;
        const local_ordinal_type r2 = part2packrowidx0_sub(partidx,local_subpartidx)+2;

#ifdef IFPACK2_BLOCKTRIDICONTAINER_USE_PRINTF
        printf("Copy for Schur complement part id = %d from kps1 = %ld to r1 = %d and from kps2 = %ld to r2 = %d partidx = %d local_subpartidx = %d;\n", packidx, kps1, r1, kps2, r2, partidx, local_subpartidx);
#endif

        // Need to copy D to e_internal_vector_values.
        copy3DView<local_ordinal_type>(member, Kokkos::subview(e_internal_vector_values, 0, r1, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL()), 
                          Kokkos::subview(internal_vector_values, kps1, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL()));

        copy3DView<local_ordinal_type>(member, Kokkos::subview(e_internal_vector_values, 1, r2, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL()), 
                          Kokkos::subview(internal_vector_values, kps2, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL()));

      }
      
      KOKKOS_INLINE_FUNCTION
      void
      operator() (const ComputeETag &, const member_type &member) const {
        // btdm is packed and sorted from largest one
        const local_ordinal_type packidx = packindices_sub(member.league_rank());

        const local_ordinal_type subpartidx = packptr_sub(packidx);
        const local_ordinal_type n_parts = part2packrowidx0_sub.extent(0);
        const local_ordinal_type local_subpartidx = subpartidx/n_parts;
        const local_ordinal_type partidx = subpartidx%n_parts;

        const local_ordinal_type npacks = packptr_sub(packidx+1) - subpartidx;
        const local_ordinal_type i0 = pack_td_ptr(partidx,local_subpartidx);
        const local_ordinal_type r0 = part2packrowidx0_sub(partidx,local_subpartidx);
        const local_ordinal_type nrows = partptr_sub(subpartidx,1) - partptr_sub(subpartidx,0);
        const local_ordinal_type num_vectors = blocksize;

        (void) npacks;

        internal_vector_scratch_type_3d_view
          WW(member.team_scratch(0), blocksize, num_vectors, vector_loop_size);
        if (local_subpartidx == 0) {
          Kokkos::parallel_for
            (Kokkos::ThreadVectorRange(member, vector_loop_size),[&](const int &v) {
              solveMultiVector<impl_type, internal_vector_scratch_type_3d_view> (member, blocksize, i0, r0, nrows, v, internal_vector_values, Kokkos::subview(e_internal_vector_values, 0, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL()), WW, true);
            });
        }
        else if (local_subpartidx == (local_ordinal_type) part2packrowidx0_sub.extent(1) - 2) {
          Kokkos::parallel_for
            (Kokkos::ThreadVectorRange(member, vector_loop_size),[&](const int &v) {
              solveMultiVector<impl_type, internal_vector_scratch_type_3d_view> (member, blocksize, i0, r0, nrows, v, internal_vector_values, Kokkos::subview(e_internal_vector_values, 1, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL()), WW);
            });
        }
        else {
          Kokkos::parallel_for
            (Kokkos::ThreadVectorRange(member, vector_loop_size),[&](const int &v) {
              solveMultiVector<impl_type, internal_vector_scratch_type_3d_view> (member, blocksize, i0, r0, nrows, v, internal_vector_values, Kokkos::subview(e_internal_vector_values, 0, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL()), WW, true);
              solveMultiVector<impl_type, internal_vector_scratch_type_3d_view> (member, blocksize, i0, r0, nrows, v, internal_vector_values, Kokkos::subview(e_internal_vector_values, 1, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL()), WW); 
            });
        }
      }

      KOKKOS_INLINE_FUNCTION
      void
      operator() (const ComputeSchurTag &, const member_type &member) const {
        // btdm is packed and sorted from largest one
        const local_ordinal_type packindices_schur_i = member.league_rank() % packindices_schur.extent(0);
        const local_ordinal_type packindices_schur_j = member.league_rank() / packindices_schur.extent(0);
        const local_ordinal_type packidx = packindices_schur(packindices_schur_i, packindices_schur_j);

        const local_ordinal_type subpartidx = packptr_sub(packidx);
        const local_ordinal_type n_parts = part2packrowidx0_sub.extent(0);
        const local_ordinal_type local_subpartidx = subpartidx/n_parts;
        const local_ordinal_type partidx = subpartidx%n_parts;

        //const local_ordinal_type npacks = packptr_sub(packidx+1) - subpartidx;
        const local_ordinal_type i0 = pack_td_ptr(partidx,local_subpartidx);
        //const local_ordinal_type r0 = part2packrowidx0_sub(partidx,local_subpartidx);
        //const local_ordinal_type nrows = partptr_sub(subpartidx,1) - partptr_sub(subpartidx,0);

        internal_vector_scratch_type_3d_view
          WW(member.team_scratch(0), blocksize, blocksize, vector_loop_size);

        // Compute S = D - C E

        const local_ordinal_type local_subpartidx_schur = (local_subpartidx-1)/2;
        const local_ordinal_type i0_schur = local_subpartidx_schur == 0 ? pack_td_ptr_schur(partidx,local_subpartidx_schur) : pack_td_ptr_schur(partidx,local_subpartidx_schur) + 1;
        const local_ordinal_type i0_offset = local_subpartidx_schur == 0 ? i0+2 : i0+2;

        for  (local_ordinal_type i = 0; i < 4; ++i) { //pack_td_ptr_schur(partidx,local_subpartidx_schur+1)-i0_schur
          copy3DView<local_ordinal_type>(member, Kokkos::subview(internal_vector_values_schur, i0_schur+i, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL()), 
                            Kokkos::subview(internal_vector_values, i0_offset+i, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL()));
        }

        member.team_barrier();

        const auto one = Kokkos::ArithTraits<btdm_magnitude_type>::one();

        const size_type c_kps1 = pack_td_ptr(partidx, local_subpartidx)+1;
        const size_type c_kps2 = pack_td_ptr(partidx, local_subpartidx+1)-2;

        const local_ordinal_type e_r1 = part2packrowidx0_sub(partidx,local_subpartidx)-1;
        const local_ordinal_type e_r2 = part2packrowidx0_sub(partidx,local_subpartidx)+2;

        typedef ExtractAndFactorizeTridiagsDefaultModeAndAlgo
          <typename execution_space::memory_space> default_mode_and_algo_type;

        typedef typename default_mode_and_algo_type::mode_type default_mode_type;
        typedef typename default_mode_and_algo_type::algo_type default_algo_type;

        Kokkos::parallel_for
          (Kokkos::ThreadVectorRange(member, vector_loop_size),[&](const int &v) {
            for  (size_type i = 0; i < pack_td_ptr_schur(partidx,local_subpartidx_schur+1)-pack_td_ptr_schur(partidx,local_subpartidx_schur); ++i) {
              local_ordinal_type e_r, e_c, c_kps;

              if ( local_subpartidx_schur == 0 ) {
                if ( i == 0 ) {
                  e_r = e_r1;
                  e_c = 0;
                  c_kps = c_kps1;
                }
                else if ( i == 3 ) {
                  e_r = e_r2;
                  e_c = 1;
                  c_kps = c_kps2;
                }
                else if ( i == 4 ) {
                  e_r = e_r2;
                  e_c = 0;
                  c_kps = c_kps2;
                }
                else {
                  continue;
                }
              }
              else {
                if ( i == 0 ) {
                  e_r = e_r1;
                  e_c = 1;
                  c_kps = c_kps1;
                }
                else if ( i == 1 ) {
                  e_r = e_r1;
                  e_c = 0;
                  c_kps = c_kps1;
                }
                else if ( i == 4 ) {
                  e_r = e_r2;
                  e_c = 1;
                  c_kps = c_kps2;
                }
                else if ( i == 5 ) {
                  e_r = e_r2;
                  e_c = 0;
                  c_kps = c_kps2;
                }
                else {
                  continue;
                }
              }

              auto S = Kokkos::subview(internal_vector_values_schur, pack_td_ptr_schur(partidx,local_subpartidx_schur)+i, Kokkos::ALL(), Kokkos::ALL(), v);
              auto C = Kokkos::subview(internal_vector_values, c_kps, Kokkos::ALL(), Kokkos::ALL(), v);
              auto E = Kokkos::subview(e_internal_vector_values, e_c, e_r, Kokkos::ALL(), Kokkos::ALL(), v);
              KB::Gemm<member_type,
                      KB::Trans::NoTranspose,KB::Trans::NoTranspose,
                      default_mode_type,default_algo_type>
                ::invoke(member, -one, C, E, one, S);
            }
          });
      }

      KOKKOS_INLINE_FUNCTION
      void
      operator() (const FactorizeSchurTag &, const member_type &member) const {
        const local_ordinal_type packidx = packindices_schur(member.league_rank(), 0);

        const local_ordinal_type subpartidx = packptr_sub(packidx);

        const local_ordinal_type n_parts = part2packrowidx0_sub.extent(0);
        const local_ordinal_type partidx = subpartidx%n_parts;

        const local_ordinal_type i0 = pack_td_ptr_schur(partidx,0);
        const local_ordinal_type nrows = 2*(pack_td_ptr_schur.extent(1)-1);

        internal_vector_scratch_type_3d_view
          WW(member.team_scratch(0), blocksize, blocksize, vector_loop_size);
        
#ifdef IFPACK2_BLOCKTRIDICONTAINER_USE_PRINTF
        printf("FactorizeSchurTag rank = %d, i0 = %d, nrows = %d, vector_loop_size = %d;\n", member.league_rank(), i0, nrows, vector_loop_size);
#endif

        if (vector_loop_size == 1) {
          factorize_subline(member, i0, nrows, 0, internal_vector_values_schur, WW);
        } else {
          Kokkos::parallel_for
            (Kokkos::ThreadVectorRange(member, vector_loop_size),
	     [&](const local_ordinal_type &v) {
              factorize_subline(member, i0, nrows, v, internal_vector_values_schur, WW);
            });
        }
      }

      void run() {
        IFPACK2_BLOCKTRIDICONTAINER_PROFILER_REGION_BEGIN;
        const local_ordinal_type team_size =
          ExtractAndFactorizeTridiagsDefaultModeAndAlgo<typename execution_space::memory_space>::
          recommended_team_size(blocksize, vector_length, internal_vector_length);
        const local_ordinal_type per_team_scratch = internal_vector_scratch_type_3d_view::
          shmem_size(blocksize, blocksize, vector_loop_size);

        {
#ifdef IFPACK2_BLOCKTRIDICONTAINER_USE_PRINTF
        printf("Start ExtractAndFactorizeSubLineTag\n");
#endif
          IFPACK2_BLOCKHELPER_TIMER("BlockTriDi::NumericPhase::ExtractAndFactorizeSubLineTag", ExtractAndFactorizeSubLineTag0);
          Kokkos::TeamPolicy<execution_space,ExtractAndFactorizeSubLineTag>
            policy(packindices_sub.extent(0), team_size, vector_loop_size);


          const local_ordinal_type n_parts = part2packrowidx0_sub.extent(0);
          writeBTDValuesToFile(n_parts, scalar_values, "before.mm");

          policy.set_scratch_size(0,Kokkos::PerTeam(per_team_scratch));
          Kokkos::parallel_for("ExtractAndFactorize::TeamPolicy::run<ExtractAndFactorizeSubLineTag>",
                              policy, *this);
          execution_space().fence();

          writeBTDValuesToFile(n_parts, scalar_values, "after.mm");
#ifdef IFPACK2_BLOCKTRIDICONTAINER_USE_PRINTF
        printf("End ExtractAndFactorizeSubLineTag\n");
#endif
        }

        if (packindices_schur.extent(1) > 0)
        {
          {
#ifdef IFPACK2_BLOCKTRIDICONTAINER_USE_PRINTF
        printf("Start ExtractBCDTag\n");
#endif
            Kokkos::deep_copy(e_scalar_values, Kokkos::ArithTraits<btdm_magnitude_type>::zero());
            Kokkos::deep_copy(scalar_values_schur, Kokkos::ArithTraits<btdm_magnitude_type>::zero());

            write5DMultiVectorValuesToFile(part2packrowidx0_sub.extent(0), e_scalar_values, "e_scalar_values_before_extract.mm");

            {
              IFPACK2_BLOCKHELPER_TIMER("BlockTriDi::NumericPhase::ExtractBCDTag", ExtractBCDTag0);
              Kokkos::TeamPolicy<execution_space,ExtractBCDTag>
                policy(packindices_schur.extent(0)*packindices_schur.extent(1), team_size, vector_loop_size);

              policy.set_scratch_size(0,Kokkos::PerTeam(per_team_scratch));
              Kokkos::parallel_for("ExtractAndFactorize::TeamPolicy::run<ExtractBCDTag>",
                                  policy, *this);
              execution_space().fence();
            }

#ifdef IFPACK2_BLOCKTRIDICONTAINER_USE_PRINTF
        printf("End ExtractBCDTag\n");
#endif
            writeBTDValuesToFile(part2packrowidx0_sub.extent(0), scalar_values, "after_extraction_of_BCD.mm");
#ifdef IFPACK2_BLOCKTRIDICONTAINER_USE_PRINTF
        printf("Start ComputeETag\n");
#endif
            write5DMultiVectorValuesToFile(part2packrowidx0_sub.extent(0), e_scalar_values, "e_scalar_values_after_extract.mm");
            {
              IFPACK2_BLOCKHELPER_TIMER("BlockTriDi::NumericPhase::ComputeETag", ComputeETag0);
              Kokkos::TeamPolicy<execution_space,ComputeETag>
                policy(packindices_sub.extent(0), team_size, vector_loop_size);

              policy.set_scratch_size(0,Kokkos::PerTeam(per_team_scratch));
              Kokkos::parallel_for("ExtractAndFactorize::TeamPolicy::run<ComputeETag>",
                                  policy, *this);
              execution_space().fence();
            }
            write5DMultiVectorValuesToFile(part2packrowidx0_sub.extent(0), e_scalar_values, "e_scalar_values_after_compute.mm");

#ifdef IFPACK2_BLOCKTRIDICONTAINER_USE_PRINTF
        printf("End ComputeETag\n");
#endif
          }

          {
#ifdef IFPACK2_BLOCKTRIDICONTAINER_USE_PRINTF
        printf("Start ComputeSchurTag\n");
#endif
            IFPACK2_BLOCKHELPER_TIMER("BlockTriDi::NumericPhase::ComputeSchurTag", ComputeSchurTag0);
            writeBTDValuesToFile(part2packrowidx0_sub.extent(0), scalar_values_schur, "before_schur.mm");
            Kokkos::TeamPolicy<execution_space,ComputeSchurTag>
              policy(packindices_schur.extent(0)*packindices_schur.extent(1), team_size, vector_loop_size);

            policy.set_scratch_size(0,Kokkos::PerTeam(per_team_scratch));
            Kokkos::parallel_for("ExtractAndFactorize::TeamPolicy::run<ComputeSchurTag>",
                                policy, *this);
            writeBTDValuesToFile(part2packrowidx0_sub.extent(0), scalar_values_schur, "after_schur.mm");
            execution_space().fence();
#ifdef IFPACK2_BLOCKTRIDICONTAINER_USE_PRINTF
        printf("End ComputeSchurTag\n");
#endif
          }

          {
#ifdef IFPACK2_BLOCKTRIDICONTAINER_USE_PRINTF
        printf("Start FactorizeSchurTag\n");
#endif
            IFPACK2_BLOCKHELPER_TIMER("BlockTriDi::NumericPhase::FactorizeSchurTag", FactorizeSchurTag0);
            Kokkos::TeamPolicy<execution_space,FactorizeSchurTag>
              policy(packindices_schur.extent(0), team_size, vector_loop_size);
            policy.set_scratch_size(0,Kokkos::PerTeam(per_team_scratch));
            Kokkos::parallel_for("ExtractAndFactorize::TeamPolicy::run<FactorizeSchurTag>",
                                policy, *this);
            execution_space().fence();
            writeBTDValuesToFile(part2packrowidx0_sub.extent(0), scalar_values_schur, "after_factor_schur.mm");
#ifdef IFPACK2_BLOCKTRIDICONTAINER_USE_PRINTF
        printf("End FactorizeSchurTag\n");
#endif
          }
        }

        IFPACK2_BLOCKTRIDICONTAINER_PROFILER_REGION_END;
      }

    };

    ///
    /// top level numeric interface
    ///
    template<typename MatrixType>
    void
    performNumericPhase(const Teuchos::RCP<const typename BlockHelperDetails::ImplType<MatrixType>::tpetra_row_matrix_type> &A,
                        const Teuchos::RCP<const typename BlockHelperDetails::ImplType<MatrixType>::tpetra_crs_graph_type> &G,
                        const BlockHelperDetails::PartInterface<MatrixType> &interf,
                        BlockTridiags<MatrixType> &btdm,
                        const typename BlockHelperDetails::ImplType<MatrixType>::magnitude_type tiny) {
      IFPACK2_BLOCKHELPER_TIMER("BlockTriDi::NumericPhase", NumericPhase);
      ExtractAndFactorizeTridiags<MatrixType> function(btdm, interf, A, G, tiny);
      function.run();
      IFPACK2_BLOCKHELPER_TIMER_FENCE(typename BlockHelperDetails::ImplType<MatrixType>::execution_space)
    }

    ///
    /// pack multivector
    ///
    template<typename MatrixType>
    struct MultiVectorConverter {
    public:
      using impl_type = BlockHelperDetails::ImplType<MatrixType>;
      using execution_space = typename impl_type::execution_space;
      using memory_space = typename impl_type::memory_space;

      using local_ordinal_type = typename impl_type::local_ordinal_type;
      using impl_scalar_type = typename impl_type::impl_scalar_type;
      using btdm_scalar_type = typename impl_type::btdm_scalar_type;
      using tpetra_multivector_type = typename impl_type::tpetra_multivector_type;
      using local_ordinal_type_1d_view = typename impl_type::local_ordinal_type_1d_view;
      using vector_type_3d_view = typename impl_type::vector_type_3d_view;
      using impl_scalar_type_2d_view_tpetra = typename impl_type::impl_scalar_type_2d_view_tpetra;
      using const_impl_scalar_type_2d_view_tpetra = typename impl_scalar_type_2d_view_tpetra::const_type;
      static constexpr int vector_length = impl_type::vector_length;

      using member_type = typename Kokkos::TeamPolicy<execution_space>::member_type;

    private:
      // part interface
      const ConstUnmanaged<local_ordinal_type_1d_view> partptr;
      const ConstUnmanaged<local_ordinal_type_1d_view> packptr;
      const ConstUnmanaged<local_ordinal_type_1d_view> part2packrowidx0;
      const ConstUnmanaged<local_ordinal_type_1d_view> part2rowidx0;
      const ConstUnmanaged<local_ordinal_type_1d_view> lclrow;
      const local_ordinal_type blocksize;
      const local_ordinal_type num_vectors;

      // packed multivector output (or input)
      vector_type_3d_view packed_multivector;
      const_impl_scalar_type_2d_view_tpetra scalar_multivector;

      template<typename TagType>
      KOKKOS_INLINE_FUNCTION
      void copy_multivectors(const local_ordinal_type &j,
                             const local_ordinal_type &vi,
                             const local_ordinal_type &pri,
                             const local_ordinal_type &ri0) const {
        for (local_ordinal_type col=0;col<num_vectors;++col)
          for (local_ordinal_type i=0;i<blocksize;++i)
            packed_multivector(pri, i, col)[vi] = static_cast<btdm_scalar_type>(scalar_multivector(blocksize*lclrow(ri0+j)+i,col));
      }

    public:

      MultiVectorConverter(const BlockHelperDetails::PartInterface<MatrixType> &interf,
                           const vector_type_3d_view &pmv)
        : partptr(interf.partptr),
          packptr(interf.packptr),
          part2packrowidx0(interf.part2packrowidx0),
          part2rowidx0(interf.part2rowidx0),
          lclrow(interf.lclrow),
          blocksize(pmv.extent(1)),
          num_vectors(pmv.extent(2)),
          packed_multivector(pmv) {}

      // TODO:: modify this routine similar to the team level functions
      // inline  ---> FIXME HIP: should not need the KOKKOS_INLINE_FUNCTION below...
      KOKKOS_INLINE_FUNCTION
      void
      operator() (const local_ordinal_type &packidx) const {
        local_ordinal_type partidx = packptr(packidx);
        local_ordinal_type npacks = packptr(packidx+1) - partidx;
        const local_ordinal_type pri0 = part2packrowidx0(partidx);

        local_ordinal_type ri0[vector_length] = {};
        local_ordinal_type nrows[vector_length] = {};
        for (local_ordinal_type v=0;v<npacks;++v,++partidx) {
          ri0[v] = part2rowidx0(partidx);
          nrows[v] = part2rowidx0(partidx+1) - ri0[v];
        }
        for (local_ordinal_type j=0;j<nrows[0];++j) {
          local_ordinal_type cnt = 1;
          for (;cnt<npacks && j!= nrows[cnt];++cnt);
          npacks = cnt;
          const local_ordinal_type pri = pri0 + j;
          for (local_ordinal_type col=0;col<num_vectors;++col)
            for (local_ordinal_type i=0;i<blocksize;++i)
              for (local_ordinal_type v=0;v<npacks;++v)
                packed_multivector(pri, i, col)[v] = static_cast<btdm_scalar_type>(scalar_multivector(blocksize*lclrow(ri0[v]+j)+i,col));
        }
      }

      KOKKOS_INLINE_FUNCTION
      void
      operator() (const member_type &member) const {
        const local_ordinal_type packidx = member.league_rank();
        const local_ordinal_type partidx_begin = packptr(packidx);
        const local_ordinal_type npacks = packptr(packidx+1) - partidx_begin;
        const local_ordinal_type pri0 = part2packrowidx0(partidx_begin);
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, npacks), [&](const local_ordinal_type &v) {
            const local_ordinal_type partidx = partidx_begin + v;
            const local_ordinal_type ri0 = part2rowidx0(partidx);
            const local_ordinal_type nrows = part2rowidx0(partidx+1) - ri0;

            if (nrows == 1) {
              const local_ordinal_type pri = pri0;
              for (local_ordinal_type col=0;col<num_vectors;++col) {
                Kokkos::parallel_for(Kokkos::TeamThreadRange(member, blocksize), [&](const local_ordinal_type &i) {
                    packed_multivector(pri, i, col)[v] = static_cast<btdm_scalar_type>(scalar_multivector(blocksize*lclrow(ri0)+i,col));
                  });
              }
            } else {
              Kokkos::parallel_for(Kokkos::TeamThreadRange(member, nrows), [&](const local_ordinal_type &j) {
                  const local_ordinal_type pri = pri0 + j;
                  for (local_ordinal_type col=0;col<num_vectors;++col)
                    for (local_ordinal_type i=0;i<blocksize;++i)
                      packed_multivector(pri, i, col)[v] = static_cast<btdm_scalar_type>(scalar_multivector(blocksize*lclrow(ri0+j)+i,col));
                });
            }
          });
      }

      void run(const const_impl_scalar_type_2d_view_tpetra &scalar_multivector_) {
        IFPACK2_BLOCKTRIDICONTAINER_PROFILER_REGION_BEGIN;
        IFPACK2_BLOCKHELPER_TIMER("BlockTriDi::MultiVectorConverter", MultiVectorConverter0);

        scalar_multivector = scalar_multivector_;
        if constexpr (BlockHelperDetails::is_device<execution_space>::value) {
          const local_ordinal_type vl = vector_length;
          const Kokkos::TeamPolicy<execution_space> policy(packptr.extent(0) - 1, Kokkos::AUTO(), vl);
          Kokkos::parallel_for
            ("MultiVectorConverter::TeamPolicy", policy, *this);
        } else {
          const Kokkos::RangePolicy<execution_space> policy(0, packptr.extent(0) - 1);
          Kokkos::parallel_for
            ("MultiVectorConverter::RangePolicy", policy, *this);
        }
        IFPACK2_BLOCKTRIDICONTAINER_PROFILER_REGION_END;
        IFPACK2_BLOCKHELPER_TIMER_FENCE(execution_space)
      }
    };

    ///
    /// solve tridiags
    ///

    template<>
    struct SolveTridiagsDefaultModeAndAlgo<Kokkos::HostSpace> {
      typedef KB::Mode::Serial mode_type;
      typedef KB::Algo::Level2::Unblocked single_vector_algo_type;
#if defined(__KOKKOSBATCHED_INTEL_MKL_COMPACT_BATCHED__)
      typedef KB::Algo::Level3::CompactMKL multi_vector_algo_type;
#else
      typedef KB::Algo::Level3::Blocked multi_vector_algo_type;
#endif
      static int recommended_team_size(const int /* blksize */,
                                       const int /* vector_length */,
                                       const int /* internal_vector_length */) {
        return 1;
      }
    };

#if defined(KOKKOS_ENABLE_CUDA)
    static inline int SolveTridiagsRecommendedCudaTeamSize(const int blksize,
                                                           const int vector_length,
                                                           const int internal_vector_length) {
      const int vector_size = vector_length/internal_vector_length;
      int total_team_size(0);
      if      (blksize <=  5) total_team_size =  32;
      else if (blksize <=  9) total_team_size =  32; // 64
      else if (blksize <= 12) total_team_size =  96;
      else if (blksize <= 16) total_team_size = 128;
      else if (blksize <= 20) total_team_size = 160;
      else                    total_team_size = 160;
      return total_team_size/vector_size;
    }

    template<>
    struct SolveTridiagsDefaultModeAndAlgo<Kokkos::CudaSpace> {
      typedef KB::Mode::Team mode_type;
      typedef KB::Algo::Level2::Unblocked single_vector_algo_type;
      typedef KB::Algo::Level3::Unblocked multi_vector_algo_type;
      static int recommended_team_size(const int blksize,
                                       const int vector_length,
                                       const int internal_vector_length) {
        return SolveTridiagsRecommendedCudaTeamSize(blksize, vector_length, internal_vector_length);
      }
    };
    template<>
    struct SolveTridiagsDefaultModeAndAlgo<Kokkos::CudaUVMSpace> {
      typedef KB::Mode::Team mode_type;
      typedef KB::Algo::Level2::Unblocked single_vector_algo_type;
      typedef KB::Algo::Level3::Unblocked multi_vector_algo_type;
      static int recommended_team_size(const int blksize,
                                       const int vector_length,
                                       const int internal_vector_length) {
        return SolveTridiagsRecommendedCudaTeamSize(blksize, vector_length, internal_vector_length);
      }
    };
#endif

#if defined(KOKKOS_ENABLE_HIP)
    static inline int SolveTridiagsRecommendedHIPTeamSize(const int blksize,
							  const int vector_length,
							  const int internal_vector_length) {
      const int vector_size = vector_length/internal_vector_length;
      int total_team_size(0);
      if      (blksize <=  5) total_team_size =  32;
      else if (blksize <=  9) total_team_size =  32; // 64
      else if (blksize <= 12) total_team_size =  96;
      else if (blksize <= 16) total_team_size = 128;
      else if (blksize <= 20) total_team_size = 160;
      else                    total_team_size = 160;
      return total_team_size/vector_size;
    }

    template<>
    struct SolveTridiagsDefaultModeAndAlgo<Kokkos::HIPSpace> {
      typedef KB::Mode::Team mode_type;
      typedef KB::Algo::Level2::Unblocked single_vector_algo_type;
      typedef KB::Algo::Level3::Unblocked multi_vector_algo_type;
      static int recommended_team_size(const int blksize,
                                       const int vector_length,
                                       const int internal_vector_length) {
        return SolveTridiagsRecommendedHIPTeamSize(blksize, vector_length, internal_vector_length);
      }
    };
    template<>
    struct SolveTridiagsDefaultModeAndAlgo<Kokkos::HIPHostPinnedSpace> {
      typedef KB::Mode::Team mode_type;
      typedef KB::Algo::Level2::Unblocked single_vector_algo_type;
      typedef KB::Algo::Level3::Unblocked multi_vector_algo_type;
      static int recommended_team_size(const int blksize,
                                       const int vector_length,
                                       const int internal_vector_length) {
        return SolveTridiagsRecommendedHIPTeamSize(blksize, vector_length, internal_vector_length);
      }
    };
#endif

#if defined(KOKKOS_ENABLE_SYCL)
    static inline int SolveTridiagsRecommendedSYCLTeamSize(const int blksize,
                                                          const int vector_length,
                                                          const int internal_vector_length) {
      const int vector_size = vector_length/internal_vector_length;
      int total_team_size(0);
      if      (blksize <=  5) total_team_size =  32;
      else if (blksize <=  9) total_team_size =  32; // 64                                                                                                                                                          
      else if (blksize <= 12) total_team_size =  96;
      else if (blksize <= 16) total_team_size = 128;
      else if (blksize <= 20) total_team_size = 160;
      else                    total_team_size = 160;
      return total_team_size/vector_size;
    }

    template<>
    struct SolveTridiagsDefaultModeAndAlgo<Kokkos::Experimental::SYCLSharedUSMSpace> {
      typedef KB::Mode::Team mode_type;
      typedef KB::Algo::Level2::Unblocked single_vector_algo_type;
      typedef KB::Algo::Level3::Unblocked multi_vector_algo_type;
      static int recommended_team_size(const int blksize,
                                       const int vector_length,
                                       const int internal_vector_length) {
        return SolveTridiagsRecommendedSYCLTeamSize(blksize, vector_length, internal_vector_length);
      }
    };
    template<>
    struct SolveTridiagsDefaultModeAndAlgo<Kokkos::Experimental::SYCLDeviceUSMSpace> {
      typedef KB::Mode::Team mode_type;
      typedef KB::Algo::Level2::Unblocked single_vector_algo_type;
      typedef KB::Algo::Level3::Unblocked multi_vector_algo_type;
      static int recommended_team_size(const int blksize,
                                       const int vector_length,
                                       const int internal_vector_length) {
        return SolveTridiagsRecommendedSYCLTeamSize(blksize, vector_length, internal_vector_length);
      }
    };
#endif



    
    template<typename MatrixType>
    struct SolveTridiags {
    public:
      using impl_type = BlockHelperDetails::ImplType<MatrixType>;
      using execution_space = typename impl_type::execution_space;

      using local_ordinal_type = typename impl_type::local_ordinal_type;
      using size_type = typename impl_type::size_type;
      using impl_scalar_type = typename impl_type::impl_scalar_type;
      using magnitude_type = typename impl_type::magnitude_type;
      using btdm_scalar_type = typename impl_type::btdm_scalar_type;
      using btdm_magnitude_type = typename impl_type::btdm_magnitude_type;
      /// views
      using local_ordinal_type_1d_view = typename impl_type::local_ordinal_type_1d_view;
      using local_ordinal_type_2d_view = typename impl_type::local_ordinal_type_2d_view;
      using size_type_2d_view = typename impl_type::size_type_2d_view;
      /// vectorization
      using vector_type_3d_view = typename impl_type::vector_type_3d_view;
      using internal_vector_type_4d_view = typename impl_type::internal_vector_type_4d_view;
      using internal_vector_type_5d_view = typename impl_type::internal_vector_type_5d_view;
      using btdm_scalar_type_4d_view = typename impl_type::btdm_scalar_type_4d_view;

      using internal_vector_scratch_type_3d_view = Scratch<typename impl_type::internal_vector_type_3d_view>;

      using internal_vector_type =typename impl_type::internal_vector_type;
      static constexpr int vector_length = impl_type::vector_length;
      static constexpr int internal_vector_length = impl_type::internal_vector_length;

      /// multivector view
      using impl_scalar_type_1d_view = typename impl_type::impl_scalar_type_1d_view;
      using impl_scalar_type_2d_view_tpetra = typename impl_type::impl_scalar_type_2d_view_tpetra;

      /// team policy types
      using team_policy_type = Kokkos::TeamPolicy<execution_space>;
      using member_type = typename team_policy_type::member_type;

    private:
      // part interface
      local_ordinal_type n_subparts_per_part;
      const ConstUnmanaged<local_ordinal_type_1d_view> partptr;
      const ConstUnmanaged<local_ordinal_type_1d_view> packptr;
      const ConstUnmanaged<local_ordinal_type_1d_view> packindices_sub;
      const ConstUnmanaged<local_ordinal_type_2d_view> packindices_schur;
      const ConstUnmanaged<local_ordinal_type_1d_view> part2packrowidx0;
      const ConstUnmanaged<local_ordinal_type_2d_view> part2packrowidx0_sub;
      const ConstUnmanaged<local_ordinal_type_1d_view> lclrow;
      const ConstUnmanaged<local_ordinal_type_1d_view> packptr_sub;

      const ConstUnmanaged<local_ordinal_type_2d_view> partptr_sub;
      const ConstUnmanaged<size_type_2d_view> pack_td_ptr_schur;

      // block tridiags
      const ConstUnmanaged<size_type_2d_view> pack_td_ptr;

      // block tridiags values
      const ConstUnmanaged<internal_vector_type_4d_view> D_internal_vector_values;
      const Unmanaged<internal_vector_type_4d_view> X_internal_vector_values;
      const Unmanaged<btdm_scalar_type_4d_view> X_internal_scalar_values;

      internal_vector_type_4d_view X_internal_vector_values_schur;

      const ConstUnmanaged<internal_vector_type_4d_view> D_internal_vector_values_schur;
      const ConstUnmanaged<internal_vector_type_5d_view> e_internal_vector_values;


      const local_ordinal_type vector_loop_size;

      // copy to multivectors : damping factor and Y_scalar_multivector
      Unmanaged<impl_scalar_type_2d_view_tpetra> Y_scalar_multivector;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__) || defined(__SYCL_DEVICE_ONLY__)
      AtomicUnmanaged<impl_scalar_type_1d_view> Z_scalar_vector;
#else
      /* */ Unmanaged<impl_scalar_type_1d_view> Z_scalar_vector;
#endif
      const impl_scalar_type df;
      const bool compute_diff;

    public:
      SolveTridiags(const BlockHelperDetails::PartInterface<MatrixType> &interf,
                    const BlockTridiags<MatrixType> &btdm,
                    const vector_type_3d_view &pmv,
                    const impl_scalar_type damping_factor,
                    const bool is_norm_manager_active)
        :
        // interface
        n_subparts_per_part(interf.n_subparts_per_part),
        partptr(interf.partptr),
        packptr(interf.packptr),
        packindices_sub(interf.packindices_sub),
        packindices_schur(interf.packindices_schur),
        part2packrowidx0(interf.part2packrowidx0),
        part2packrowidx0_sub(interf.part2packrowidx0_sub),
        lclrow(interf.lclrow),
        packptr_sub(interf.packptr_sub),
        partptr_sub(interf.partptr_sub),
        pack_td_ptr_schur(btdm.pack_td_ptr_schur),
        // block tridiags and  multivector
        pack_td_ptr(btdm.pack_td_ptr),
        D_internal_vector_values((internal_vector_type*)btdm.values.data(),
                                 btdm.values.extent(0),
                                 btdm.values.extent(1),
                                 btdm.values.extent(2),
                                 vector_length/internal_vector_length),
        X_internal_vector_values((internal_vector_type*)pmv.data(),
                                 pmv.extent(0),
                                 pmv.extent(1),
                                 pmv.extent(2),
                                 vector_length/internal_vector_length),
        X_internal_scalar_values((btdm_scalar_type*)pmv.data(),
                                 pmv.extent(0),
                                 pmv.extent(1),
                                 pmv.extent(2),
                                 vector_length),
        X_internal_vector_values_schur(do_not_initialize_tag("X_internal_vector_values_schur"),
                                       2*(n_subparts_per_part-1) * part2packrowidx0_sub.extent(0),
                                       pmv.extent(1),
                                       pmv.extent(2),
                                       vector_length/internal_vector_length),
        D_internal_vector_values_schur((internal_vector_type*)btdm.values_schur.data(),
                               btdm.values_schur.extent(0),
                               btdm.values_schur.extent(1),
                               btdm.values_schur.extent(2),
                               vector_length/internal_vector_length),
        e_internal_vector_values((internal_vector_type*)btdm.e_values.data(),
                                btdm.e_values.extent(0),
                                btdm.e_values.extent(1),
                                btdm.e_values.extent(2),
                                btdm.e_values.extent(3),
                                vector_length/internal_vector_length),
        vector_loop_size(vector_length/internal_vector_length),
        Y_scalar_multivector(),
        Z_scalar_vector(),
        df(damping_factor),
        compute_diff(is_norm_manager_active)
      {}

    public:

      /// move packed multi vector into flat multi vector for computing residuals
      KOKKOS_INLINE_FUNCTION
      void
      copyToFlatMultiVector(const member_type &member,
                            const local_ordinal_type partidxbeg, // partidx for v = 0
                            const local_ordinal_type npacks,
                            const local_ordinal_type pri0,
                            const local_ordinal_type v, // index with a loop of vector_loop_size
                            const local_ordinal_type blocksize,
                            const local_ordinal_type num_vectors) const {
        const local_ordinal_type vbeg = v*internal_vector_length;
        if (vbeg < npacks) {
          local_ordinal_type ri0_vals[internal_vector_length] = {};
          local_ordinal_type nrows_vals[internal_vector_length] = {};
          for (local_ordinal_type vv=vbeg,vi=0;vv<npacks && vi<internal_vector_length;++vv,++vi) {
            const local_ordinal_type partidx = partidxbeg+vv;
            ri0_vals[vi] = partptr(partidx);
            nrows_vals[vi] = partptr(partidx+1) - ri0_vals[vi];
          }

          impl_scalar_type z_partial_sum(0);
          if (nrows_vals[0] == 1) {
            const local_ordinal_type j=0, pri=pri0;
            {
              for (local_ordinal_type vv=vbeg,vi=0;vv<npacks && vi<internal_vector_length;++vv,++vi) {
                const local_ordinal_type ri0 = ri0_vals[vi];
                const local_ordinal_type nrows = nrows_vals[vi];
                if (j < nrows) {
                  Kokkos::parallel_for
                    (Kokkos::TeamThreadRange(member, blocksize),
                     [&](const local_ordinal_type &i) {
                      const local_ordinal_type row = blocksize*lclrow(ri0+j)+i;
                      for (local_ordinal_type col=0;col<num_vectors;++col) {
                        impl_scalar_type &y = Y_scalar_multivector(row,col);
                        const impl_scalar_type yd = X_internal_vector_values(pri, i, col, v)[vi] - y;
                        y  += yd;

                        {//if (compute_diff) {
                          const auto yd_abs = Kokkos::ArithTraits<impl_scalar_type>::abs(yd);
                          z_partial_sum += yd_abs*yd_abs;
                        }
                      }
                    });
                }
              }
            }
          } else {
            Kokkos::parallel_for
              (Kokkos::TeamThreadRange(member, nrows_vals[0]),
               [&](const local_ordinal_type &j) {
                const local_ordinal_type pri = pri0 + j;
                for (local_ordinal_type vv=vbeg,vi=0;vv<npacks && vi<internal_vector_length;++vv,++vi) {
                  const local_ordinal_type ri0 = ri0_vals[vi];
                  const local_ordinal_type nrows = nrows_vals[vi];
                  if (j < nrows) {
                    for (local_ordinal_type col=0;col<num_vectors;++col) {
                      for (local_ordinal_type i=0;i<blocksize;++i) {
                        const local_ordinal_type row = blocksize*lclrow(ri0+j)+i;
                        impl_scalar_type &y = Y_scalar_multivector(row,col);
                        const impl_scalar_type yd = X_internal_vector_values(pri, i, col, v)[vi] - y;
                        y += yd;

                        {//if (compute_diff) {
                          const auto yd_abs = Kokkos::ArithTraits<impl_scalar_type>::abs(yd);
                          z_partial_sum += yd_abs*yd_abs;
                        }
                      }
                    }
                  }
                }
              });
          }
          //if (compute_diff)
            Z_scalar_vector(member.league_rank()) += z_partial_sum;
        }
      }

      ///
      /// cuda team vectorization
      ///
      template<typename WWViewType>
      KOKKOS_INLINE_FUNCTION
      void
      solveSingleVector(const member_type &member,
                        const local_ordinal_type &blocksize,
                        const local_ordinal_type &i0,
                        const local_ordinal_type &r0,
                        const local_ordinal_type &nrows,
                        const local_ordinal_type &v,
                        const WWViewType &WW) const {

        typedef SolveTridiagsDefaultModeAndAlgo
          <typename execution_space::memory_space> default_mode_and_algo_type;

        typedef typename default_mode_and_algo_type::mode_type default_mode_type;
        typedef typename default_mode_and_algo_type::single_vector_algo_type default_algo_type;

        // base pointers
        auto A = D_internal_vector_values.data();
        auto X = X_internal_vector_values.data();

        // constant
        const auto one = Kokkos::ArithTraits<btdm_magnitude_type>::one();
        const auto zero = Kokkos::ArithTraits<btdm_magnitude_type>::zero();
        //const local_ordinal_type num_vectors = X_scalar_values.extent(2);

        // const local_ordinal_type blocksize = D_scalar_values.extent(1);
        const local_ordinal_type astep = D_internal_vector_values.stride_0();
        const local_ordinal_type as0 = D_internal_vector_values.stride_1(); //blocksize*vector_length;
        const local_ordinal_type as1 = D_internal_vector_values.stride_2(); //vector_length;
        const local_ordinal_type xstep = X_internal_vector_values.stride_0();
        const local_ordinal_type xs0 = X_internal_vector_values.stride_1(); //vector_length;

        // move to starting point
        A += i0*astep + v;
        X += r0*xstep + v;

        //for (local_ordinal_type col=0;col<num_vectors;++col)
        if (nrows > 1) {
          // solve Lx = x
          KOKKOSBATCHED_TRSV_LOWER_NO_TRANSPOSE_INTERNAL_INVOKE
            (default_mode_type,default_algo_type,
             member,
             KB::Diag::Unit,
             blocksize,blocksize,
             one,
             A, as0, as1,
             X, xs0);

          for (local_ordinal_type tr=1;tr<nrows;++tr) {
            member.team_barrier();
            KOKKOSBATCHED_GEMV_NO_TRANSPOSE_INTERNAL_INVOKE
              (default_mode_type,default_algo_type,
               member,
               blocksize, blocksize,
               -one,
               A+2*astep, as0, as1,
               X, xs0,
               one,
               X+1*xstep, xs0);
            KOKKOSBATCHED_TRSV_LOWER_NO_TRANSPOSE_INTERNAL_INVOKE
              (default_mode_type,default_algo_type,
               member,
               KB::Diag::Unit,
               blocksize,blocksize,
               one,
               A+3*astep, as0, as1,
               X+1*xstep, xs0);

            A += 3*astep;
            X += 1*xstep;
          }

          // solve Ux = x
          KOKKOSBATCHED_TRSV_UPPER_NO_TRANSPOSE_INTERNAL_INVOKE
            (default_mode_type,default_algo_type,
             member,
             KB::Diag::NonUnit,
             blocksize, blocksize,
             one,
             A, as0, as1,
             X, xs0);

          for (local_ordinal_type tr=nrows;tr>1;--tr) {
            A -= 3*astep;
            member.team_barrier();
            KOKKOSBATCHED_GEMV_NO_TRANSPOSE_INTERNAL_INVOKE
              (default_mode_type,default_algo_type,
               member,
               blocksize, blocksize,
               -one,
               A+1*astep, as0, as1,
               X, xs0,
               one,
               X-1*xstep, xs0);
            KOKKOSBATCHED_TRSV_UPPER_NO_TRANSPOSE_INTERNAL_INVOKE
              (default_mode_type,default_algo_type,
               member,
               KB::Diag::NonUnit,
               blocksize, blocksize,
               one,
               A, as0, as1,
               X-1*xstep,xs0);
            X -= 1*xstep;
          }
          // for multiple rhs
          //X += xs1;
        } else {
          const local_ordinal_type ws0 = WW.stride_0();
          auto W = WW.data() + v;
          KOKKOSBATCHED_COPY_VECTOR_NO_TRANSPOSE_INTERNAL_INVOKE
            (default_mode_type,
             member, blocksize, X, xs0, W, ws0);
          member.team_barrier();
          KOKKOSBATCHED_GEMV_NO_TRANSPOSE_INTERNAL_INVOKE
            (default_mode_type,default_algo_type,
             member,
             blocksize, blocksize,
             one,
             A, as0, as1,
             W, xs0,
             zero,
             X, xs0);
        }
      }

      template<typename WWViewType>
      KOKKOS_INLINE_FUNCTION
      void
      solveMultiVector(const member_type &member,
                       const local_ordinal_type &/* blocksize */,
                       const local_ordinal_type &i0,
                       const local_ordinal_type &r0,
                       const local_ordinal_type &nrows,
                       const local_ordinal_type &v,
                       const WWViewType &WW) const {

        typedef SolveTridiagsDefaultModeAndAlgo
          <typename execution_space::memory_space> default_mode_and_algo_type;

        typedef typename default_mode_and_algo_type::mode_type default_mode_type;
        typedef typename default_mode_and_algo_type::multi_vector_algo_type default_algo_type;

        // constant
        const auto one = Kokkos::ArithTraits<btdm_magnitude_type>::one();
        const auto zero = Kokkos::ArithTraits<btdm_magnitude_type>::zero();

        // subview pattern
        auto A  = Kokkos::subview(D_internal_vector_values, i0, Kokkos::ALL(), Kokkos::ALL(), v);
        auto X1 = Kokkos::subview(X_internal_vector_values, r0, Kokkos::ALL(), Kokkos::ALL(), v);
        auto X2 = X1;

        local_ordinal_type i = i0, r = r0;


        if (nrows > 1) {
          // solve Lx = x
          KB::Trsm<member_type,
                   KB::Side::Left,KB::Uplo::Lower,KB::Trans::NoTranspose,KB::Diag::Unit,
                   default_mode_type,default_algo_type>
            ::invoke(member, one, A, X1);
          for (local_ordinal_type tr=1;tr<nrows;++tr,i+=3) {
            A.assign_data( &D_internal_vector_values(i+2,0,0,v) );
            X2.assign_data( &X_internal_vector_values(++r,0,0,v) );
            member.team_barrier();
            KB::Gemm<member_type,
                     KB::Trans::NoTranspose,KB::Trans::NoTranspose,
                     default_mode_type,default_algo_type>
              ::invoke(member, -one, A, X1, one, X2);
            A.assign_data( &D_internal_vector_values(i+3,0,0,v) );
            KB::Trsm<member_type,
                     KB::Side::Left,KB::Uplo::Lower,KB::Trans::NoTranspose,KB::Diag::Unit,
                     default_mode_type,default_algo_type>
              ::invoke(member, one, A, X2);
            X1.assign_data( X2.data() );
          }

          // solve Ux = x
          KB::Trsm<member_type,
                   KB::Side::Left,KB::Uplo::Upper,KB::Trans::NoTranspose,KB::Diag::NonUnit,
                   default_mode_type,default_algo_type>
            ::invoke(member, one, A, X1);
          for (local_ordinal_type tr=nrows;tr>1;--tr) {
            i -= 3;
            A.assign_data( &D_internal_vector_values(i+1,0,0,v) );
            X2.assign_data( &X_internal_vector_values(--r,0,0,v) );
            member.team_barrier();
            KB::Gemm<member_type,
                     KB::Trans::NoTranspose,KB::Trans::NoTranspose,
                     default_mode_type,default_algo_type>
              ::invoke(member, -one, A, X1, one, X2);

            A.assign_data( &D_internal_vector_values(i,0,0,v) );
            KB::Trsm<member_type,
                     KB::Side::Left,KB::Uplo::Upper,KB::Trans::NoTranspose,KB::Diag::NonUnit,
                     default_mode_type,default_algo_type>
              ::invoke(member, one, A, X2);
            X1.assign_data( X2.data() );
          }
        } else {
          // matrix is already inverted
          auto W = Kokkos::subview(WW, Kokkos::ALL(), Kokkos::ALL(), v);
          KB::Copy<member_type,KB::Trans::NoTranspose,default_mode_type>
            ::invoke(member, X1, W);
          member.team_barrier();
          KB::Gemm<member_type,
                   KB::Trans::NoTranspose,KB::Trans::NoTranspose,
                   default_mode_type,default_algo_type>
            ::invoke(member, one, A, W, zero, X1);
        }
      }

      template<int B> struct SingleVectorTag {};
      template<int B> struct MultiVectorTag {};

      template<int B> struct SingleVectorSubLineTag {};
      template<int B> struct MultiVectorSubLineTag {};
      template<int B> struct SingleVectorApplyCTag {};
      template<int B> struct MultiVectorApplyCTag {};
      template<int B> struct SingleVectorSchurTag {};
      template<int B> struct MultiVectorSchurTag {};
      template<int B> struct SingleVectorApplyETag {};
      template<int B> struct MultiVectorApplyETag {};
      template<int B> struct SingleVectorCopyToFlatTag {};
      template<int B> struct SingleZeroingTag {};

      template<int B>
      KOKKOS_INLINE_FUNCTION
      void
      operator() (const SingleVectorTag<B> &, const member_type &member) const {
        const local_ordinal_type packidx = member.league_rank();
        const local_ordinal_type partidx = packptr(packidx);
        const local_ordinal_type npacks = packptr(packidx+1) - partidx;
        const local_ordinal_type pri0 = part2packrowidx0(partidx);
        const local_ordinal_type i0 = pack_td_ptr(partidx,0);
        const local_ordinal_type r0 = part2packrowidx0(partidx);
        const local_ordinal_type nrows = partptr(partidx+1) - partptr(partidx);
        const local_ordinal_type blocksize = (B == 0 ? D_internal_vector_values.extent(1) : B);
        const local_ordinal_type num_vectors = 1;
        internal_vector_scratch_type_3d_view
          WW(member.team_scratch(0), blocksize, 1, vector_loop_size);
        Kokkos::single(Kokkos::PerTeam(member), [&]() {
            Z_scalar_vector(member.league_rank()) = impl_scalar_type(0);
          });
        Kokkos::parallel_for
          (Kokkos::ThreadVectorRange(member, vector_loop_size),[&](const int &v) {
            solveSingleVector(member, blocksize, i0, r0, nrows, v, WW);
            copyToFlatMultiVector(member, partidx, npacks, pri0, v, blocksize, num_vectors);
          });
      }

      template<int B>
      KOKKOS_INLINE_FUNCTION
      void
      operator() (const MultiVectorTag<B> &, const member_type &member) const {
        const local_ordinal_type packidx = member.league_rank();
        const local_ordinal_type partidx = packptr(packidx);
        const local_ordinal_type npacks = packptr(packidx+1) - partidx;
        const local_ordinal_type pri0 = part2packrowidx0(partidx);
        const local_ordinal_type i0 = pack_td_ptr(partidx,0);
        const local_ordinal_type r0 = part2packrowidx0(partidx);
        const local_ordinal_type nrows = partptr(partidx+1) - partptr(partidx);
        const local_ordinal_type blocksize = (B == 0 ? D_internal_vector_values.extent(1) : B);
        const local_ordinal_type num_vectors = X_internal_vector_values.extent(2);

        internal_vector_scratch_type_3d_view
          WW(member.team_scratch(0), blocksize, num_vectors, vector_loop_size);
        Kokkos::single(Kokkos::PerTeam(member), [&]() {
            Z_scalar_vector(member.league_rank()) = impl_scalar_type(0);
          });
        Kokkos::parallel_for
          (Kokkos::ThreadVectorRange(member, vector_loop_size),[&](const int &v) {
            solveMultiVector(member, blocksize, i0, r0, nrows, v, WW);
            copyToFlatMultiVector(member, partidx, npacks, pri0, v, blocksize, num_vectors);
          });
      }

      template<int B>
      KOKKOS_INLINE_FUNCTION
      void
      operator() (const SingleVectorSubLineTag<B> &, const member_type &member) const {
        // btdm is packed and sorted from largest one
        const local_ordinal_type packidx = packindices_sub(member.league_rank());

        const local_ordinal_type subpartidx = packptr_sub(packidx);
        const local_ordinal_type n_parts = part2packrowidx0_sub.extent(0);
        const local_ordinal_type local_subpartidx = subpartidx/n_parts;
        const local_ordinal_type partidx = subpartidx%n_parts;

        const local_ordinal_type npacks = packptr_sub(packidx+1) - subpartidx;
        const local_ordinal_type i0 = pack_td_ptr(partidx,local_subpartidx);
        const local_ordinal_type r0 = part2packrowidx0_sub(partidx,local_subpartidx);
        const local_ordinal_type nrows = partptr_sub(subpartidx,1) - partptr_sub(subpartidx,0);
        const local_ordinal_type blocksize = e_internal_vector_values.extent(2);

        //(void) i0;
        //(void) nrows;
        (void) npacks;

        internal_vector_scratch_type_3d_view
          WW(member.team_scratch(0), blocksize, 1, vector_loop_size);

        Kokkos::parallel_for
          (Kokkos::ThreadVectorRange(member, vector_loop_size),[&](const int &v) {
            solveSingleVectorNew<impl_type, internal_vector_scratch_type_3d_view> (member, blocksize, i0, r0, nrows, v, D_internal_vector_values, X_internal_vector_values, WW);
          });
      }

      template<int B>
      KOKKOS_INLINE_FUNCTION
      void
      operator() (const SingleVectorApplyCTag<B> &, const member_type &member) const {
        // btdm is packed and sorted from largest one
        //const local_ordinal_type packidx = packindices_schur(member.league_rank());
        const local_ordinal_type packidx = packindices_sub(member.league_rank());

        const local_ordinal_type subpartidx = packptr_sub(packidx);
        const local_ordinal_type n_parts = part2packrowidx0_sub.extent(0);
        const local_ordinal_type local_subpartidx = subpartidx/n_parts;
        const local_ordinal_type partidx = subpartidx%n_parts;
        const local_ordinal_type blocksize = e_internal_vector_values.extent(2);

        //const local_ordinal_type npacks = packptr_sub(packidx+1) - subpartidx;
        const local_ordinal_type i0 = pack_td_ptr(partidx,local_subpartidx);
        const local_ordinal_type r0 = part2packrowidx0_sub(partidx,local_subpartidx);
        const local_ordinal_type nrows = partptr_sub(subpartidx,1) - partptr_sub(subpartidx,0);

        internal_vector_scratch_type_3d_view
          WW(member.team_scratch(0), blocksize, blocksize, vector_loop_size);

        // Compute v_2 = v_2 - C v_1

        const local_ordinal_type local_subpartidx_schur = (local_subpartidx-1)/2;
        const local_ordinal_type i0_schur = local_subpartidx_schur == 0 ? pack_td_ptr_schur(partidx,local_subpartidx_schur) : pack_td_ptr_schur(partidx,local_subpartidx_schur) + 1;
        const local_ordinal_type i0_offset = local_subpartidx_schur == 0 ? i0+2 : i0+2;

        (void) i0_schur;
        (void) i0_offset;

        const auto one = Kokkos::ArithTraits<btdm_magnitude_type>::one();

        const size_type c_kps2 =  local_subpartidx > 0 ? pack_td_ptr(partidx, local_subpartidx)-2 : 0;
        const size_type c_kps1 = pack_td_ptr(partidx, local_subpartidx+1)+1;

        typedef SolveTridiagsDefaultModeAndAlgo
          <typename execution_space::memory_space> default_mode_and_algo_type;

        typedef typename default_mode_and_algo_type::mode_type default_mode_type;
        typedef typename default_mode_and_algo_type::single_vector_algo_type default_algo_type;

        if (local_subpartidx == 0) {
          Kokkos::parallel_for
            (Kokkos::ThreadVectorRange(member, vector_loop_size),[&](const int &v) {
              auto v_1 = Kokkos::subview(X_internal_vector_values, r0+nrows-1, Kokkos::ALL(), 0, v);
              auto v_2 = Kokkos::subview(X_internal_vector_values, r0+nrows, Kokkos::ALL(), 0, v);
              auto C = Kokkos::subview(D_internal_vector_values, c_kps1, Kokkos::ALL(), Kokkos::ALL(), v);

              KOKKOSBATCHED_GEMV_NO_TRANSPOSE_INTERNAL_INVOKE
                (default_mode_type,default_algo_type,
                  member,
                  blocksize, blocksize,
                  -one,
                  C.data(), C.stride_0(), C.stride_1(),
                  v_1.data(), v_1.stride_0(),
                  one,
                  v_2.data(), v_2.stride_0());
            });
        }
        else if (local_subpartidx == (local_ordinal_type) part2packrowidx0_sub.extent(1) - 2) {
          Kokkos::parallel_for
            (Kokkos::ThreadVectorRange(member, vector_loop_size),[&](const int &v) {
              auto v_1 = Kokkos::subview(X_internal_vector_values, r0, Kokkos::ALL(), 0, v);
              auto v_2 = Kokkos::subview(X_internal_vector_values, r0-1, Kokkos::ALL(), 0, v);
              auto C = Kokkos::subview(D_internal_vector_values, c_kps2, Kokkos::ALL(), Kokkos::ALL(), v);

              KOKKOSBATCHED_GEMV_NO_TRANSPOSE_INTERNAL_INVOKE
                (default_mode_type,default_algo_type,
                  member,
                  blocksize, blocksize,
                  -one,
                  C.data(), C.stride_0(), C.stride_1(),
                  v_1.data(), v_1.stride_0(),
                  one,
                  v_2.data(), v_2.stride_0());
            });
        }
        else {
          Kokkos::parallel_for
            (Kokkos::ThreadVectorRange(member, vector_loop_size),[&](const int &v) {
              {
                auto v_1 = Kokkos::subview(X_internal_vector_values, r0+nrows-1, Kokkos::ALL(), 0, v);
                auto v_2 = Kokkos::subview(X_internal_vector_values, r0+nrows, Kokkos::ALL(), 0, v);
                auto C = Kokkos::subview(D_internal_vector_values, c_kps1, Kokkos::ALL(), Kokkos::ALL(), v);

                KOKKOSBATCHED_GEMV_NO_TRANSPOSE_INTERNAL_INVOKE
                  (default_mode_type,default_algo_type,
                    member,
                    blocksize, blocksize,
                    -one,
                    C.data(), C.stride_0(), C.stride_1(),
                    v_1.data(), v_1.stride_0(),
                    one,
                    v_2.data(), v_2.stride_0());
              }
              {
                auto v_1 = Kokkos::subview(X_internal_vector_values, r0, Kokkos::ALL(), 0, v);
                auto v_2 = Kokkos::subview(X_internal_vector_values, r0-1, Kokkos::ALL(), 0, v);
                auto C = Kokkos::subview(D_internal_vector_values, c_kps2, Kokkos::ALL(), Kokkos::ALL(), v);

                KOKKOSBATCHED_GEMV_NO_TRANSPOSE_INTERNAL_INVOKE
                  (default_mode_type,default_algo_type,
                    member,
                    blocksize, blocksize,
                    -one,
                    C.data(), C.stride_0(), C.stride_1(),
                    v_1.data(), v_1.stride_0(),
                    one,
                    v_2.data(), v_2.stride_0());
              }
            });
        }
      }

      template<int B>
      KOKKOS_INLINE_FUNCTION
      void
      operator() (const SingleVectorSchurTag<B> &, const member_type &member) const {
        const local_ordinal_type packidx = packindices_sub(member.league_rank());

        const local_ordinal_type partidx = packptr_sub(packidx);

        const local_ordinal_type blocksize = e_internal_vector_values.extent(2);

        const local_ordinal_type i0_schur = pack_td_ptr_schur(partidx,0);
        const local_ordinal_type nrows = 2*(n_subparts_per_part-1);

        const local_ordinal_type r0_schur = nrows * member.league_rank();

        internal_vector_scratch_type_3d_view
          WW(member.team_scratch(0), blocksize, blocksize, vector_loop_size);
        
        for (local_ordinal_type schur_sub_part = 0; schur_sub_part < n_subparts_per_part-1; ++schur_sub_part) {
          const local_ordinal_type r0 = part2packrowidx0_sub(partidx,2*schur_sub_part+1);
          for (local_ordinal_type i = 0; i < 2; ++i) {
            copy3DView<local_ordinal_type>(member, 
              Kokkos::subview(X_internal_vector_values_schur, r0_schur+2*schur_sub_part+i, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL()), 
              Kokkos::subview(X_internal_vector_values, r0+i, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL()));
          }
        }

        Kokkos::parallel_for
          (Kokkos::ThreadVectorRange(member, vector_loop_size),[&](const int &v) {
            solveSingleVectorNew<impl_type, internal_vector_scratch_type_3d_view> (member, blocksize, i0_schur, r0_schur, nrows, v, D_internal_vector_values_schur, X_internal_vector_values_schur, WW);
          });

        for (local_ordinal_type schur_sub_part = 0; schur_sub_part < n_subparts_per_part-1; ++schur_sub_part) {
          const local_ordinal_type r0 = part2packrowidx0_sub(partidx,2*schur_sub_part+1);
          for (local_ordinal_type i = 0; i < 2; ++i) {
            copy3DView<local_ordinal_type>(member, 
              Kokkos::subview(X_internal_vector_values, r0+i, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL()), 
              Kokkos::subview(X_internal_vector_values_schur, r0_schur+2*schur_sub_part+i, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL()));
          }
        }
      }

      template<int B>
      KOKKOS_INLINE_FUNCTION
      void
      operator() (const SingleVectorApplyETag<B> &, const member_type &member) const {
        const local_ordinal_type packidx = packindices_sub(member.league_rank());

        const local_ordinal_type subpartidx = packptr_sub(packidx);
        const local_ordinal_type n_parts = part2packrowidx0_sub.extent(0);
        const local_ordinal_type local_subpartidx = subpartidx/n_parts;
        const local_ordinal_type partidx = subpartidx%n_parts;
        const local_ordinal_type blocksize = e_internal_vector_values.extent(2);

        const local_ordinal_type r0 = part2packrowidx0_sub(partidx,local_subpartidx);
        const local_ordinal_type nrows = partptr_sub(subpartidx,1) - partptr_sub(subpartidx,0);

        internal_vector_scratch_type_3d_view
          WW(member.team_scratch(0), blocksize, blocksize, vector_loop_size);

        // Compute v_2 = v_2 - C v_1

        const auto one = Kokkos::ArithTraits<btdm_magnitude_type>::one();

        typedef SolveTridiagsDefaultModeAndAlgo
          <typename execution_space::memory_space> default_mode_and_algo_type;

        typedef typename default_mode_and_algo_type::mode_type default_mode_type;
        typedef typename default_mode_and_algo_type::single_vector_algo_type default_algo_type;

        if (local_subpartidx == 0) {
          Kokkos::parallel_for
            (Kokkos::ThreadVectorRange(member, vector_loop_size),[&](const int &v) {

              auto v_2 = Kokkos::subview(X_internal_vector_values, r0+nrows, Kokkos::ALL(), 0, v);

              for (local_ordinal_type row = 0; row < nrows; ++row) {
                auto v_1 = Kokkos::subview(X_internal_vector_values, r0+row, Kokkos::ALL(), 0, v);
                auto E = Kokkos::subview(e_internal_vector_values, 0, r0+row, Kokkos::ALL(), Kokkos::ALL(), v);

                KOKKOSBATCHED_GEMV_NO_TRANSPOSE_INTERNAL_INVOKE
                  (default_mode_type,default_algo_type,
                    member,
                    blocksize, blocksize,
                    -one,
                    E.data(), E.stride_0(), E.stride_1(),
                    v_2.data(), v_2.stride_0(),
                    one,
                    v_1.data(), v_1.stride_0());
              }
            });
        }
        else if (local_subpartidx == (local_ordinal_type) part2packrowidx0_sub.extent(1) - 2) {
          Kokkos::parallel_for
            (Kokkos::ThreadVectorRange(member, vector_loop_size),[&](const int &v) {
              auto v_2 = Kokkos::subview(X_internal_vector_values, r0-1, Kokkos::ALL(), 0, v);

              for (local_ordinal_type row = 0; row < nrows; ++row) {
                auto v_1 = Kokkos::subview(X_internal_vector_values, r0+row, Kokkos::ALL(), 0, v);
                auto E = Kokkos::subview(e_internal_vector_values, 1, r0+row, Kokkos::ALL(), Kokkos::ALL(), v);

                KOKKOSBATCHED_GEMV_NO_TRANSPOSE_INTERNAL_INVOKE
                  (default_mode_type,default_algo_type,
                    member,
                    blocksize, blocksize,
                    -one,
                    E.data(), E.stride_0(), E.stride_1(),
                    v_2.data(), v_2.stride_0(),
                    one,
                    v_1.data(), v_1.stride_0());
              }
            });
        }
        else {
          Kokkos::parallel_for
            (Kokkos::ThreadVectorRange(member, vector_loop_size),[&](const int &v) {
              {
                auto v_2 = Kokkos::subview(X_internal_vector_values, r0+nrows, Kokkos::ALL(), 0, v);

                for (local_ordinal_type row = 0; row < nrows; ++row) {
                  auto v_1 = Kokkos::subview(X_internal_vector_values, r0+row, Kokkos::ALL(), 0, v);
                  auto E = Kokkos::subview(e_internal_vector_values, 0, r0+row, Kokkos::ALL(), Kokkos::ALL(), v);

                  KOKKOSBATCHED_GEMV_NO_TRANSPOSE_INTERNAL_INVOKE
                    (default_mode_type,default_algo_type,
                      member,
                      blocksize, blocksize,
                      -one,
                      E.data(), E.stride_0(), E.stride_1(),
                      v_2.data(), v_2.stride_0(),
                      one,
                      v_1.data(), v_1.stride_0());
                }
              }
              {
                auto v_2 = Kokkos::subview(X_internal_vector_values, r0-1, Kokkos::ALL(), 0, v);

                for (local_ordinal_type row = 0; row < nrows; ++row) {
                  auto v_1 = Kokkos::subview(X_internal_vector_values, r0+row, Kokkos::ALL(), 0, v);
                  auto E = Kokkos::subview(e_internal_vector_values, 1, r0+row, Kokkos::ALL(), Kokkos::ALL(), v);

                  KOKKOSBATCHED_GEMV_NO_TRANSPOSE_INTERNAL_INVOKE
                    (default_mode_type,default_algo_type,
                      member,
                      blocksize, blocksize,
                      -one,
                      E.data(), E.stride_0(), E.stride_1(),
                      v_2.data(), v_2.stride_0(),
                      one,
                      v_1.data(), v_1.stride_0());
                }
              }
            });
        }
      }

      template<int B>
      KOKKOS_INLINE_FUNCTION
      void
      operator() (const SingleVectorCopyToFlatTag<B> &, const member_type &member) const {
        const local_ordinal_type packidx = member.league_rank();
        const local_ordinal_type partidx = packptr(packidx);
        const local_ordinal_type npacks = packptr(packidx+1) - partidx;
        const local_ordinal_type pri0 = part2packrowidx0(partidx);
        const local_ordinal_type blocksize = (B == 0 ? D_internal_vector_values.extent(1) : B);
        const local_ordinal_type num_vectors = 1;

        Kokkos::parallel_for
          (Kokkos::ThreadVectorRange(member, vector_loop_size),[&](const int &v) {
            copyToFlatMultiVector(member, partidx, npacks, pri0, v, blocksize, num_vectors);
          });
      }    

      template<int B>
      KOKKOS_INLINE_FUNCTION
      void
      operator() (const SingleZeroingTag<B> &, const member_type &member) const {
        Kokkos::single(Kokkos::PerTeam(member), [&]() {
            Z_scalar_vector(member.league_rank()) = impl_scalar_type(0);
          });
      }

      void run(const impl_scalar_type_2d_view_tpetra &Y,
               const impl_scalar_type_1d_view &Z) {
        IFPACK2_BLOCKTRIDICONTAINER_PROFILER_REGION_BEGIN;
        IFPACK2_BLOCKHELPER_TIMER("BlockTriDi::SolveTridiags", SolveTridiags);

        /// set vectors
        this->Y_scalar_multivector = Y;
        this->Z_scalar_vector = Z;

        const local_ordinal_type num_vectors = X_internal_vector_values.extent(2);
        const local_ordinal_type blocksize = D_internal_vector_values.extent(1);

        const local_ordinal_type team_size =
          SolveTridiagsDefaultModeAndAlgo<typename execution_space::memory_space>::
          recommended_team_size(blocksize, vector_length, internal_vector_length);
        const int per_team_scratch = internal_vector_scratch_type_3d_view
          ::shmem_size(blocksize, num_vectors, vector_loop_size);

#if defined(KOKKOS_ENABLE_DEPRECATED_CODE)
#define BLOCKTRIDICONTAINER_DETAILS_SOLVETRIDIAGS(B)                    \
        if (num_vectors == 1) {                                         \
          const Kokkos::TeamPolicy<execution_space,SingleVectorTag<B> > \
            policy(packptr.extent(0) - 1, team_size, vector_loop_size); \
          Kokkos::parallel_for                                          \
            ("SolveTridiags::TeamPolicy::run<SingleVector>",            \
             policy.set_scratch_size(0,Kokkos::PerTeam(per_team_scratch)), *this); \
        } else {                                                        \
          const Kokkos::TeamPolicy<execution_space,MultiVectorTag<B> > \
            policy(packptr.extent(0) - 1, team_size, vector_loop_size); \
          Kokkos::parallel_for                                          \
            ("SolveTridiags::TeamPolicy::run<MultiVector>",             \
             policy.set_scratch_size(0,Kokkos::PerTeam(per_team_scratch)), *this); \
        } break
#else
#define BLOCKTRIDICONTAINER_DETAILS_SOLVETRIDIAGS(B)                    \
        if (num_vectors == 1) {                                         \
          if (packindices_schur.extent(1) <= 0) { \
            Kokkos::TeamPolicy<execution_space,SingleVectorTag<B> >       \
              policy(packptr.extent(0) - 1, team_size, vector_loop_size); \
            policy.set_scratch_size(0,Kokkos::PerTeam(per_team_scratch)); \
            Kokkos::parallel_for                                          \
              ("SolveTridiags::TeamPolicy::run<SingleVector>",            \
              policy, *this);                                            \
          } \
          else { \
            { \
               \
              Kokkos::TeamPolicy<execution_space,SingleZeroingTag<B> >       \
                policy(packptr.extent(0) - 1, team_size, vector_loop_size); \
              Kokkos::parallel_for                                          \
                ("SolveTridiags::TeamPolicy::run<SingleZeroingTag>",            \
                policy, *this);                                            \
            } \
            { \
              IFPACK2_BLOCKHELPER_TIMER("BlockTriDi::ApplyInverseJacobi::SingleVectorSubLineTag", SingleVectorSubLineTag0); \
              write4DMultiVectorValuesToFile(part2packrowidx0_sub.extent(0), X_internal_scalar_values, "x_scalar_values_before_SingleVectorSubLineTag.mm"); \
              Kokkos::TeamPolicy<execution_space,SingleVectorSubLineTag<B> >       \
                policy(packindices_sub.extent(0), team_size, vector_loop_size); \
              policy.set_scratch_size(0,Kokkos::PerTeam(per_team_scratch)); \
              Kokkos::parallel_for                                          \
                ("SolveTridiags::TeamPolicy::run<SingleVector>",            \
                policy, *this);                                            \
              write4DMultiVectorValuesToFile(part2packrowidx0_sub.extent(0), X_internal_scalar_values, "x_scalar_values_after_SingleVectorSubLineTag.mm"); \
              IFPACK2_BLOCKHELPER_TIMER_FENCE(execution_space) \
            } \
            { \
              IFPACK2_BLOCKHELPER_TIMER("BlockTriDi::ApplyInverseJacobi::SingleVectorApplyCTag", SingleVectorApplyCTag0); \
              write4DMultiVectorValuesToFile(part2packrowidx0_sub.extent(0), X_internal_scalar_values, "x_scalar_values_before_SingleVectorApplyCTag.mm"); \
              Kokkos::TeamPolicy<execution_space,SingleVectorApplyCTag<B> >       \
                policy(packindices_sub.extent(0), team_size, vector_loop_size); \
              policy.set_scratch_size(0,Kokkos::PerTeam(per_team_scratch)); \
              Kokkos::parallel_for                                          \
                ("SolveTridiags::TeamPolicy::run<SingleVector>",            \
                policy, *this);                                            \
              write4DMultiVectorValuesToFile(part2packrowidx0_sub.extent(0), X_internal_scalar_values, "x_scalar_values_after_SingleVectorApplyCTag.mm"); \
              IFPACK2_BLOCKHELPER_TIMER_FENCE(execution_space) \
            } \
            { \
              IFPACK2_BLOCKHELPER_TIMER("BlockTriDi::ApplyInverseJacobi::SingleVectorSchurTag", SingleVectorSchurTag0); \
              write4DMultiVectorValuesToFile(part2packrowidx0_sub.extent(0), X_internal_scalar_values, "x_scalar_values_before_SingleVectorSchurTag.mm"); \
              Kokkos::TeamPolicy<execution_space,SingleVectorSchurTag<B> >       \
                policy(packindices_schur.extent(0), team_size, vector_loop_size); \
              policy.set_scratch_size(0,Kokkos::PerTeam(per_team_scratch)); \
              Kokkos::parallel_for                                          \
                ("SolveTridiags::TeamPolicy::run<SingleVector>",            \
                policy, *this);                                            \
              write4DMultiVectorValuesToFile(part2packrowidx0_sub.extent(0), X_internal_scalar_values, "x_scalar_values_after_SingleVectorSchurTag.mm"); \
              IFPACK2_BLOCKHELPER_TIMER_FENCE(execution_space) \
            } \
            { \
              IFPACK2_BLOCKHELPER_TIMER("BlockTriDi::ApplyInverseJacobi::SingleVectorApplyETag", SingleVectorApplyETag0); \
              write4DMultiVectorValuesToFile(part2packrowidx0_sub.extent(0), X_internal_scalar_values, "x_scalar_values_before_SingleVectorApplyETag.mm"); \
              Kokkos::TeamPolicy<execution_space,SingleVectorApplyETag<B> >       \
                policy(packindices_sub.extent(0), team_size, vector_loop_size); \
              policy.set_scratch_size(0,Kokkos::PerTeam(per_team_scratch)); \
              Kokkos::parallel_for                                          \
                ("SolveTridiags::TeamPolicy::run<SingleVector>",            \
                policy, *this);                                            \
              write4DMultiVectorValuesToFile(part2packrowidx0_sub.extent(0), X_internal_scalar_values, "x_scalar_values_after_SingleVectorApplyETag.mm"); \
              IFPACK2_BLOCKHELPER_TIMER_FENCE(execution_space) \
            } \
            { \
               \
              Kokkos::TeamPolicy<execution_space,SingleVectorCopyToFlatTag<B> >       \
                policy(packptr.extent(0) - 1, team_size, vector_loop_size); \
              Kokkos::parallel_for                                          \
                ("SolveTridiags::TeamPolicy::run<SingleVectorCopyToFlatTag>",            \
                policy, *this);                                            \
            } \
          } \
        } else {                                                        \
          Kokkos::TeamPolicy<execution_space,MultiVectorTag<B> >        \
            policy(packptr.extent(0) - 1, team_size, vector_loop_size); \
          policy.set_scratch_size(0,Kokkos::PerTeam(per_team_scratch)); \
          Kokkos::parallel_for                                          \
            ("SolveTridiags::TeamPolicy::run<MultiVector>",             \
             policy, *this);                                            \
        } break
#endif
        switch (blocksize) {
        case   3: BLOCKTRIDICONTAINER_DETAILS_SOLVETRIDIAGS( 3);
        case   5: BLOCKTRIDICONTAINER_DETAILS_SOLVETRIDIAGS( 5);
        case   6: BLOCKTRIDICONTAINER_DETAILS_SOLVETRIDIAGS( 6);
        case   7: BLOCKTRIDICONTAINER_DETAILS_SOLVETRIDIAGS( 7);
        case  10: BLOCKTRIDICONTAINER_DETAILS_SOLVETRIDIAGS(10);
        case  11: BLOCKTRIDICONTAINER_DETAILS_SOLVETRIDIAGS(11);
        case  12: BLOCKTRIDICONTAINER_DETAILS_SOLVETRIDIAGS(12);
        case  13: BLOCKTRIDICONTAINER_DETAILS_SOLVETRIDIAGS(13);
        case  16: BLOCKTRIDICONTAINER_DETAILS_SOLVETRIDIAGS(16);
        case  17: BLOCKTRIDICONTAINER_DETAILS_SOLVETRIDIAGS(17);
        case  18: BLOCKTRIDICONTAINER_DETAILS_SOLVETRIDIAGS(18);
        case  19: BLOCKTRIDICONTAINER_DETAILS_SOLVETRIDIAGS(19);
        default : BLOCKTRIDICONTAINER_DETAILS_SOLVETRIDIAGS( 0);
        }
#undef BLOCKTRIDICONTAINER_DETAILS_SOLVETRIDIAGS

        IFPACK2_BLOCKTRIDICONTAINER_PROFILER_REGION_END;
        IFPACK2_BLOCKHELPER_TIMER_FENCE(execution_space)
      }
    };

    ///
    /// top level apply interface
    ///
    template<typename MatrixType>
    int
    applyInverseJacobi(// importer
                       const Teuchos::RCP<const typename BlockHelperDetails::ImplType<MatrixType>::tpetra_row_matrix_type> &A,
                       // tpetra interface
                       const typename BlockHelperDetails::ImplType<MatrixType>::tpetra_multivector_type &X,  // tpetra interface
                       /* */ typename BlockHelperDetails::ImplType<MatrixType>::tpetra_multivector_type &Y,  // tpetra interface
                       /* */ typename BlockHelperDetails::ImplType<MatrixType>::tpetra_multivector_type &Z,  // temporary tpetra interface (seq_method)
                       /* */ typename BlockHelperDetails::ImplType<MatrixType>::impl_scalar_type_1d_view &W,  // temporary tpetra interface (diff)
                       // local object interface
                       const BlockHelperDetails::PartInterface<MatrixType> &interf, // mesh interface
                       const BlockTridiags<MatrixType> &btdm, // packed block tridiagonal matrices
                       /* */ typename BlockHelperDetails::ImplType<MatrixType>::vector_type_1d_view &work, // workspace for packed multivector of right hand side
                       /* */ BlockHelperDetails::NormManager<MatrixType> &norm_manager,
                       // preconditioner parameters
                       const typename BlockHelperDetails::ImplType<MatrixType>::impl_scalar_type &damping_factor,
                       /* */ bool is_y_zero,
                       const int max_num_sweeps,
                       const typename BlockHelperDetails::ImplType<MatrixType>::magnitude_type tol,
                       const int check_tol_every) {
      IFPACK2_BLOCKHELPER_TIMER("BlockTriDi::ApplyInverseJacobi", ApplyInverseJacobi);

      using impl_type = BlockHelperDetails::ImplType<MatrixType>;
      using local_ordinal_type = typename impl_type::local_ordinal_type;
      using size_type = typename impl_type::size_type;
      using impl_scalar_type = typename impl_type::impl_scalar_type;
      using magnitude_type = typename impl_type::magnitude_type;
      using vector_type_1d_view = typename impl_type::vector_type_1d_view;
      using vector_type_3d_view = typename impl_type::vector_type_3d_view;

      using impl_scalar_type_1d_view = typename impl_type::impl_scalar_type_1d_view;

      // max number of sweeps should be positive number
      TEUCHOS_TEST_FOR_EXCEPT_MSG(max_num_sweeps <= 0,
                                  "Maximum number of sweeps must be >= 1.");

      // const parameters
      const bool is_norm_manager_active = tol > Kokkos::ArithTraits<magnitude_type>::zero();
      const magnitude_type tolerance = tol*tol;
      const local_ordinal_type blocksize = btdm.values.extent(1);
      const local_ordinal_type num_vectors = Y.getNumVectors();
      const local_ordinal_type num_blockrows = interf.part2packrowidx0_back;

      const impl_scalar_type zero(0.0);
      const impl_scalar_type one(1.0);
      const impl_scalar_type mone = impl_scalar_type(-one);

      // if workspace is needed more, resize it
      const size_type work_span_required = num_blockrows*num_vectors*blocksize;
      if (work.span() < work_span_required)
        work = vector_type_1d_view("vector workspace 1d view", work_span_required);

      // construct W
      const local_ordinal_type W_size = interf.packptr.extent(0)-1;
      if (local_ordinal_type(W.extent(0)) < W_size)
        W = impl_scalar_type_1d_view("W", W_size);

      // wrap the workspace with 3d view
      vector_type_3d_view pmv(work.data(), num_blockrows, blocksize, num_vectors);
      const auto XX = X.getLocalViewDevice(Tpetra::Access::ReadOnly);
      const auto YY = Y.getLocalViewDevice(Tpetra::Access::ReadWrite);
      const auto ZZ = Z.getLocalViewDevice(Tpetra::Access::ReadWrite);
      if (is_y_zero) Kokkos::deep_copy(YY, zero);

      MultiVectorConverter<MatrixType> multivector_converter(interf, pmv);
      SolveTridiags<MatrixType> solve_tridiags(interf, btdm, pmv,
                                               damping_factor, is_norm_manager_active);

      // norm manager workspace resize
      if (is_norm_manager_active)
        norm_manager.setCheckFrequency(check_tol_every);

      Z = Tpetra::createCopy(Y);
    
      // iterate
      int sweep = 0;
      for (;sweep<max_num_sweeps;++sweep) {
        {
          if (is_y_zero) {
            // pmv := x(lclrow)
            multivector_converter.run(XX);
            Z.putScalar(zero);
          } else {
            {
              IFPACK2_BLOCKHELPER_PROFILER_REGION_BEGIN;
              IFPACK2_BLOCKHELPER_TIMER("BlockTriDi::ComputeResidual",ComputeResidual);

              // y := x - A z
              if (false) {
                {
                  IFPACK2_BLOCKHELPER_TIMER("BlockTriDi::ComputeResidual::assign",assign);
                  Y.assign(X);
                  IFPACK2_BLOCKHELPER_TIMER_FENCE(typename impl_type::execution_space)
                }
                {
                  IFPACK2_BLOCKHELPER_TIMER("BlockTriDi::ComputeResidual::apply",apply);
                  A->apply(Z, Y, Teuchos::NO_TRANS, mone, one);
                  IFPACK2_BLOCKHELPER_TIMER_FENCE(typename impl_type::execution_space)
                }
              }
              else
              {
                IFPACK2_BLOCKHELPER_TIMER("BlockTriDi::ComputeResidual::residual",residual);
                Tpetra::Details::residual(*A, Z, X, Y);
                IFPACK2_BLOCKHELPER_TIMER_FENCE(typename impl_type::execution_space)
              }

              // pmv := y(lclrow).
              {
                IFPACK2_BLOCKHELPER_TIMER("BlockTriDi::ComputeResidual::MVConverter",MVConverter);
                multivector_converter.run(YY);
                IFPACK2_BLOCKHELPER_TIMER_FENCE(typename impl_type::execution_space)
              }
              IFPACK2_BLOCKHELPER_PROFILER_REGION_END;
              IFPACK2_BLOCKHELPER_TIMER_FENCE(typename impl_type::execution_space)
            }
            if (is_norm_manager_active && norm_manager.checkDone(sweep, tolerance)) break;
          }
        }

        // pmv := inv(D) pmv.
        {
          solve_tridiags.run(YY, W);
        }

        {
          IFPACK2_BLOCKHELPER_TIMER("BlockTriDi::ComputeResidual::update",update);
          Y.update(one, Z, damping_factor);
          Z.assign(Y);
          multivector_converter.run(YY);
          IFPACK2_BLOCKHELPER_TIMER_FENCE(typename impl_type::execution_space)
        }

        {
          if (is_norm_manager_active) {
            // y(lclrow) = (b - a) y(lclrow) + a pmv, with b = 1 always.
            BlockHelperDetails::reduceVector<MatrixType>(W, norm_manager.getBuffer());
            if (sweep + 1 == max_num_sweeps) {
              norm_manager.ireduce(sweep, true);
              norm_manager.checkDone(sweep + 1, tolerance, true);
            } else {
              norm_manager.ireduce(sweep);
            }
          }
        }
        is_y_zero = false;
      }

      //sqrt the norms for the caller's use.
      if (is_norm_manager_active) norm_manager.finalize();

      return sweep;
    }


    template<typename MatrixType>
    struct ImplObject {
      using impl_type = BlockHelperDetails::ImplType<MatrixType>;
      using part_interface_type = BlockHelperDetails::PartInterface<MatrixType>;
      using block_tridiags_type = BlockTridiags<MatrixType>;
      using norm_manager_type = BlockHelperDetails::NormManager<MatrixType>;

      // distructed objects
      Teuchos::RCP<const typename impl_type::tpetra_row_matrix_type> A;
      Teuchos::RCP<const typename impl_type::tpetra_crs_graph_type> blockGraph;

      // copy of Y (mutable to penentrate const)
      mutable typename impl_type::tpetra_multivector_type Z;
      mutable typename impl_type::impl_scalar_type_1d_view W;

      // local objects
      part_interface_type part_interface;
      block_tridiags_type block_tridiags; // D
      mutable typename impl_type::vector_type_1d_view work; // right hand side workspace
      mutable norm_manager_type norm_manager;
    };

  } // namespace BlockTriDiContainerDetails

} // namespace Ifpack2

#endif
