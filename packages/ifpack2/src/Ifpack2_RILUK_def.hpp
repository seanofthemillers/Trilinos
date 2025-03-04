/*@HEADER
// ***********************************************************************
//
//       Ifpack2: Templated Object-Oriented Algebraic Preconditioner Package
//                 Copyright (2009) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// ***********************************************************************
//@HEADER
*/

#ifndef IFPACK2_CRSRILUK_DEF_HPP
#define IFPACK2_CRSRILUK_DEF_HPP

#include "Ifpack2_LocalFilter.hpp"
#include "Tpetra_CrsMatrix.hpp"
#include "Teuchos_StandardParameterEntryValidators.hpp"
#include "Ifpack2_LocalSparseTriangularSolver.hpp"
#include "Ifpack2_Details_getParamTryingTypes.hpp"
#include "Ifpack2_Details_getCrsMatrix.hpp"
#include "Kokkos_Sort.hpp"
#include "KokkosKernels_SparseUtils.hpp"
#include "KokkosKernels_Sorting.hpp"

namespace Ifpack2 {

namespace Details {
struct IlukImplType {
  enum Enum {
    Serial,
    KSPILUK   //!< Multicore/GPU KokkosKernels spiluk
  };

  static void loadPLTypeOption (Teuchos::Array<std::string>& type_strs, Teuchos::Array<Enum>& type_enums) {
    type_strs.resize(2);
    type_strs[0] = "Serial";
    type_strs[1] = "KSPILUK";
    type_enums.resize(2);
    type_enums[0] = Serial;
    type_enums[1] = KSPILUK;
  }
};
}

template<class MatrixType>
RILUK<MatrixType>::RILUK (const Teuchos::RCP<const row_matrix_type>& Matrix_in)
  : A_ (Matrix_in),
    LevelOfFill_ (0),
    Overalloc_ (2.),
    isAllocated_ (false),
    isInitialized_ (false),
    isComputed_ (false),
    numInitialize_ (0),
    numCompute_ (0),
    numApply_ (0),
    initializeTime_ (0.0),
    computeTime_ (0.0),
    applyTime_ (0.0),
    RelaxValue_ (Teuchos::ScalarTraits<magnitude_type>::zero ()),
    Athresh_ (Teuchos::ScalarTraits<magnitude_type>::zero ()),
    Rthresh_ (Teuchos::ScalarTraits<magnitude_type>::one ()),
    isKokkosKernelsSpiluk_(false)
{
  allocateSolvers();
}


template<class MatrixType>
RILUK<MatrixType>::RILUK (const Teuchos::RCP<const crs_matrix_type>& Matrix_in)
  : A_ (Matrix_in),
    LevelOfFill_ (0),
    Overalloc_ (2.),
    isAllocated_ (false),
    isInitialized_ (false),
    isComputed_ (false),
    numInitialize_ (0),
    numCompute_ (0),
    numApply_ (0),
    initializeTime_ (0.0),
    computeTime_ (0.0),
    applyTime_ (0.0),
    RelaxValue_ (Teuchos::ScalarTraits<magnitude_type>::zero ()),
    Athresh_ (Teuchos::ScalarTraits<magnitude_type>::zero ()),
    Rthresh_ (Teuchos::ScalarTraits<magnitude_type>::one ()),
    isKokkosKernelsSpiluk_(false)
{
  allocateSolvers();
}


template<class MatrixType>
RILUK<MatrixType>::~RILUK() 
{
  if (Teuchos::nonnull (KernelHandle_))
  {
    KernelHandle_->destroy_spiluk_handle();
  }
}

template<class MatrixType>
void RILUK<MatrixType>::allocateSolvers ()
{
  L_solver_ = Teuchos::rcp (new LocalSparseTriangularSolver<row_matrix_type> ());
  L_solver_->setObjectLabel("lower");
  U_solver_ = Teuchos::rcp (new LocalSparseTriangularSolver<row_matrix_type> ());
  U_solver_->setObjectLabel("upper");
}

template<class MatrixType>
void
RILUK<MatrixType>::setMatrix (const Teuchos::RCP<const row_matrix_type>& A)
{
  // It's legal for A to be null; in that case, you may not call
  // initialize() until calling setMatrix() with a nonnull input.
  // Regardless, setting the matrix invalidates any previous
  // factorization.
  if (A.getRawPtr () != A_.getRawPtr ()) {
    isAllocated_ = false;
    isInitialized_ = false;
    isComputed_ = false;
    A_local_ = Teuchos::null;
    Graph_ = Teuchos::null;

    // The sparse triangular solvers get a triangular factor as their
    // input matrix.  The triangular factors L_ and U_ are getting
    // reset, so we reset the solvers' matrices to null.  Do that
    // before setting L_ and U_ to null, so that latter step actually
    // frees the factors.
    if (! L_solver_.is_null ()) {
      L_solver_->setMatrix (Teuchos::null);
    }
    if (! U_solver_.is_null ()) {
      U_solver_->setMatrix (Teuchos::null);
    }

    L_ = Teuchos::null;
    U_ = Teuchos::null;
    D_ = Teuchos::null;
    A_ = A;
  }
}



template<class MatrixType>
const typename RILUK<MatrixType>::crs_matrix_type&
RILUK<MatrixType>::getL () const
{
  TEUCHOS_TEST_FOR_EXCEPTION(
    L_.is_null (), std::runtime_error, "Ifpack2::RILUK::getL: The L factor "
    "is null.  Please call initialize() (and preferably also compute()) "
    "before calling this method.  If the input matrix has not yet been set, "
    "you must first call setMatrix() with a nonnull input matrix before you "
    "may call initialize() or compute().");
  return *L_;
}


template<class MatrixType>
const Tpetra::Vector<typename RILUK<MatrixType>::scalar_type,
                     typename RILUK<MatrixType>::local_ordinal_type,
                     typename RILUK<MatrixType>::global_ordinal_type,
                     typename RILUK<MatrixType>::node_type>&
RILUK<MatrixType>::getD () const
{
  TEUCHOS_TEST_FOR_EXCEPTION(
    D_.is_null (), std::runtime_error, "Ifpack2::RILUK::getD: The D factor "
    "(of diagonal entries) is null.  Please call initialize() (and "
    "preferably also compute()) before calling this method.  If the input "
    "matrix has not yet been set, you must first call setMatrix() with a "
    "nonnull input matrix before you may call initialize() or compute().");
  return *D_;
}


template<class MatrixType>
const typename RILUK<MatrixType>::crs_matrix_type&
RILUK<MatrixType>::getU () const
{
  TEUCHOS_TEST_FOR_EXCEPTION(
    U_.is_null (), std::runtime_error, "Ifpack2::RILUK::getU: The U factor "
    "is null.  Please call initialize() (and preferably also compute()) "
    "before calling this method.  If the input matrix has not yet been set, "
    "you must first call setMatrix() with a nonnull input matrix before you "
    "may call initialize() or compute().");
  return *U_;
}


template<class MatrixType>
size_t RILUK<MatrixType>::getNodeSmootherComplexity() const {
  TEUCHOS_TEST_FOR_EXCEPTION(
    A_.is_null (), std::runtime_error, "Ifpack2::RILUK::getNodeSmootherComplexity: "
    "The input matrix A is null.  Please call setMatrix() with a nonnull "
    "input matrix, then call compute(), before calling this method.");
  // RILUK methods cost roughly one apply + the nnz in the upper+lower triangles
  if(!L_.is_null() && !U_.is_null())
    return A_->getLocalNumEntries() + L_->getLocalNumEntries() + U_->getLocalNumEntries();
  else
    return 0;
}


template<class MatrixType>
Teuchos::RCP<const Tpetra::Map<typename RILUK<MatrixType>::local_ordinal_type,
                               typename RILUK<MatrixType>::global_ordinal_type,
                               typename RILUK<MatrixType>::node_type> >
RILUK<MatrixType>::getDomainMap () const {
  TEUCHOS_TEST_FOR_EXCEPTION(
    A_.is_null (), std::runtime_error, "Ifpack2::RILUK::getDomainMap: "
    "The matrix is null.  Please call setMatrix() with a nonnull input "
    "before calling this method.");

  // FIXME (mfh 25 Jan 2014) Shouldn't this just come from A_?
  TEUCHOS_TEST_FOR_EXCEPTION(
    Graph_.is_null (), std::runtime_error, "Ifpack2::RILUK::getDomainMap: "
    "The computed graph is null.  Please call initialize() before calling "
    "this method.");
  return Graph_->getL_Graph ()->getDomainMap ();
}


template<class MatrixType>
Teuchos::RCP<const Tpetra::Map<typename RILUK<MatrixType>::local_ordinal_type,
                               typename RILUK<MatrixType>::global_ordinal_type,
                               typename RILUK<MatrixType>::node_type> >
RILUK<MatrixType>::getRangeMap () const {
  TEUCHOS_TEST_FOR_EXCEPTION(
    A_.is_null (), std::runtime_error, "Ifpack2::RILUK::getRangeMap: "
    "The matrix is null.  Please call setMatrix() with a nonnull input "
    "before calling this method.");

  // FIXME (mfh 25 Jan 2014) Shouldn't this just come from A_?
  TEUCHOS_TEST_FOR_EXCEPTION(
    Graph_.is_null (), std::runtime_error, "Ifpack2::RILUK::getRangeMap: "
    "The computed graph is null.  Please call initialize() before calling "
    "this method.");
  return Graph_->getL_Graph ()->getRangeMap ();
}


template<class MatrixType>
void RILUK<MatrixType>::allocate_L_and_U ()
{
  using Teuchos::null;
  using Teuchos::rcp;

  if (! isAllocated_) {
    // Deallocate any existing storage.  This avoids storing 2x
    // memory, since RCP op= does not deallocate until after the
    // assignment.
    L_ = null;
    U_ = null;
    D_ = null;

    // Allocate Matrix using ILUK graphs
    L_ = rcp (new crs_matrix_type (Graph_->getL_Graph ()));
    U_ = rcp (new crs_matrix_type (Graph_->getU_Graph ()));
    L_->setAllToScalar (STS::zero ()); // Zero out L and U matrices
    U_->setAllToScalar (STS::zero ());

    // FIXME (mfh 24 Jan 2014) This assumes domain == range Map for L and U.
    L_->fillComplete ();
    U_->fillComplete ();
    D_ = rcp (new vec_type (Graph_->getL_Graph ()->getRowMap ()));    
  }
  isAllocated_ = true;
}


template<class MatrixType>
void
RILUK<MatrixType>::
setParameters (const Teuchos::ParameterList& params)
{
  using Teuchos::RCP;
  using Teuchos::ParameterList;
  using Teuchos::Array;
  using Details::getParamTryingTypes;
  const char prefix[] = "Ifpack2::RILUK: ";

  // Default values of the various parameters.
  int fillLevel = 0;
  magnitude_type absThresh = STM::zero ();
  magnitude_type relThresh = STM::one ();
  magnitude_type relaxValue = STM::zero ();
  double overalloc = 2.;

  // "fact: iluk level-of-fill" parsing is more complicated, because
  // we want to allow as many types as make sense.  int is the native
  // type, but we also want to accept double (for backwards
  // compatibilty with ILUT).  You can't cast arbitrary magnitude_type
  // (e.g., Sacado::MP::Vector) to int, so we use float instead, to
  // get coverage of the most common magnitude_type cases.  Weirdly,
  // there's an Ifpack2 test that sets the fill level as a
  // global_ordinal_type.
  {
    const std::string paramName ("fact: iluk level-of-fill");
    getParamTryingTypes<int, int, global_ordinal_type, double, float>
      (fillLevel, params, paramName, prefix);
  }
  // For the other parameters, we prefer magnitude_type, but allow
  // double for backwards compatibility.
  {
    const std::string paramName ("fact: absolute threshold");
    getParamTryingTypes<magnitude_type, magnitude_type, double>
      (absThresh, params, paramName, prefix);
  }
  {
    const std::string paramName ("fact: relative threshold");
    getParamTryingTypes<magnitude_type, magnitude_type, double>
      (relThresh, params, paramName, prefix);
  }
  {
    const std::string paramName ("fact: relax value");
    getParamTryingTypes<magnitude_type, magnitude_type, double>
      (relaxValue, params, paramName, prefix);
  }
  {
    const std::string paramName ("fact: iluk overalloc");
    getParamTryingTypes<double, double>
      (overalloc, params, paramName, prefix);
  }

  // Parsing implementation type
  Details::IlukImplType::Enum ilukimplType = Details::IlukImplType::Serial;
  do {
    static const char typeName[] = "fact: type";

    if ( ! params.isType<std::string>(typeName)) break;

    // Map std::string <-> IlukImplType::Enum.
    Array<std::string> ilukimplTypeStrs;
    Array<Details::IlukImplType::Enum> ilukimplTypeEnums;
    Details::IlukImplType::loadPLTypeOption (ilukimplTypeStrs, ilukimplTypeEnums);
    Teuchos::StringToIntegralParameterEntryValidator<Details::IlukImplType::Enum>
      s2i(ilukimplTypeStrs (), ilukimplTypeEnums (), typeName, false);

    ilukimplType = s2i.getIntegralValue(params.get<std::string>(typeName));
  } while (0);

  if (ilukimplType == Details::IlukImplType::KSPILUK) {
    this->isKokkosKernelsSpiluk_ = true;
  }
  else {
    this->isKokkosKernelsSpiluk_ = false;
  }
  
  // Forward to trisolvers.
  L_solver_->setParameters(params);
  U_solver_->setParameters(params);

  // "Commit" the values only after validating all of them.  This
  // ensures that there are no side effects if this routine throws an
  // exception.

  LevelOfFill_ = fillLevel;
  Overalloc_ = overalloc;

  // mfh 28 Nov 2012: The previous code would not assign Athresh_,
  // Rthresh_, or RelaxValue_, if the read-in value was -1.  I don't
  // know if keeping this behavior is correct, but I'll keep it just
  // so as not to change previous behavior.

  if (absThresh != -STM::one ()) {
    Athresh_ = absThresh;
  }
  if (relThresh != -STM::one ()) {
    Rthresh_ = relThresh;
  }
  if (relaxValue != -STM::one ()) {
    RelaxValue_ = relaxValue;
  }
}


template<class MatrixType>
Teuchos::RCP<const typename RILUK<MatrixType>::row_matrix_type>
RILUK<MatrixType>::getMatrix () const {
  return Teuchos::rcp_implicit_cast<const row_matrix_type> (A_);
}


template<class MatrixType>
Teuchos::RCP<const typename RILUK<MatrixType>::crs_matrix_type>
RILUK<MatrixType>::getCrsMatrix () const {
  return Teuchos::rcp_dynamic_cast<const crs_matrix_type> (A_, true);
}


template<class MatrixType>
Teuchos::RCP<const typename RILUK<MatrixType>::row_matrix_type>
RILUK<MatrixType>::makeLocalFilter (const Teuchos::RCP<const row_matrix_type>& A)
{
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::rcp_dynamic_cast;
  using Teuchos::rcp_implicit_cast;

  // If A_'s communicator only has one process, or if its column and
  // row Maps are the same, then it is already local, so use it
  // directly.
  if (A->getRowMap ()->getComm ()->getSize () == 1 ||
      A->getRowMap ()->isSameAs (* (A->getColMap ()))) {
    return A;
  }

  // If A_ is already a LocalFilter, then use it directly.  This
  // should be the case if RILUK is being used through
  // AdditiveSchwarz, for example.
  RCP<const LocalFilter<row_matrix_type> > A_lf_r =
    rcp_dynamic_cast<const LocalFilter<row_matrix_type> > (A);
  if (! A_lf_r.is_null ()) {
    return rcp_implicit_cast<const row_matrix_type> (A_lf_r);
  }
  else {
    // A_'s communicator has more than one process, its row Map and
    // its column Map differ, and A_ is not a LocalFilter.  Thus, we
    // have to wrap it in a LocalFilter.
    return rcp (new LocalFilter<row_matrix_type> (A));
  }
}


template<class MatrixType>
void RILUK<MatrixType>::initialize ()
{
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::rcp_const_cast;
  using Teuchos::rcp_dynamic_cast;
  using Teuchos::rcp_implicit_cast;
  using Teuchos::Array;
  using Teuchos::ArrayView;
  typedef Tpetra::CrsGraph<local_ordinal_type,
                           global_ordinal_type,
                           node_type> crs_graph_type;
  const char prefix[] = "Ifpack2::RILUK::initialize: ";

  TEUCHOS_TEST_FOR_EXCEPTION
    (A_.is_null (), std::runtime_error, prefix << "The matrix is null.  Please "
     "call setMatrix() with a nonnull input before calling this method.");
  TEUCHOS_TEST_FOR_EXCEPTION
    (! A_->isFillComplete (), std::runtime_error, prefix << "The matrix is not "
     "fill complete.  You may not invoke initialize() or compute() with this "
     "matrix until the matrix is fill complete.  If your matrix is a "
     "Tpetra::CrsMatrix, please call fillComplete on it (with the domain and "
     "range Maps, if appropriate) before calling this method.");
  
  Teuchos::Time timer ("RILUK::initialize");
  double startTime = timer.wallTime();
  { // Start timing
    Teuchos::TimeMonitor timeMon (timer);

    // Calling initialize() means that the user asserts that the graph
    // of the sparse matrix may have changed.  We must not just reuse
    // the previous graph in that case.
    //
    // Regarding setting isAllocated_ to false: Eventually, we may want
    // some kind of clever memory reuse strategy, but it's always
    // correct just to blow everything away and start over.
    isInitialized_ = false;
    isAllocated_   = false;
    isComputed_    = false;
    Graph_ = Teuchos::null;

    A_local_ = makeLocalFilter (A_);
    TEUCHOS_TEST_FOR_EXCEPTION(
      A_local_.is_null (), std::logic_error, "Ifpack2::RILUK::initialize: "
      "makeLocalFilter returned null; it failed to compute A_local.  "
      "Please report this bug to the Ifpack2 developers.");

    // FIXME (mfh 24 Jan 2014, 26 Mar 2014) It would be more efficient
    // to rewrite RILUK so that it works with any RowMatrix input, not
    // just CrsMatrix.  (That would require rewriting IlukGraph to
    // handle a Tpetra::RowGraph.)  However, to make it work for now,
    // we just copy the input matrix if it's not a CrsMatrix.

    if (this->isKokkosKernelsSpiluk_) {
      this->KernelHandle_ = Teuchos::rcp (new kk_handle_type ());
      KernelHandle_->create_spiluk_handle( KokkosSparse::Experimental::SPILUKAlgorithm::SEQLVLSCHD_TP1, 
                                           A_local_->getLocalNumRows(),
                                           2*A_local_->getLocalNumEntries()*(LevelOfFill_+1), 
                                           2*A_local_->getLocalNumEntries()*(LevelOfFill_+1) );
    }

    {
      RCP<const crs_matrix_type> A_local_crs = Details::getCrsMatrix(A_local_);
      if(A_local_crs.is_null()) {
        local_ordinal_type numRows = A_local_->getLocalNumRows();
        Array<size_t> entriesPerRow(numRows);
        for(local_ordinal_type i = 0; i < numRows; i++) {
          entriesPerRow[i] = A_local_->getNumEntriesInLocalRow(i);
        }
        RCP<crs_matrix_type> A_local_crs_nc =
          rcp (new crs_matrix_type (A_local_->getRowMap (),
                                    A_local_->getColMap (),
                                    entriesPerRow()));
        // copy entries into A_local_crs
        nonconst_local_inds_host_view_type indices("indices",A_local_->getLocalMaxNumRowEntries());
        nonconst_values_host_view_type values("values",A_local_->getLocalMaxNumRowEntries());
        for(local_ordinal_type i = 0; i < numRows; i++) {
          size_t numEntries = 0;
          A_local_->getLocalRowCopy(i, indices, values, numEntries);
          A_local_crs_nc->insertLocalValues(i, numEntries, reinterpret_cast<scalar_type*>(values.data()), indices.data());
        }
        A_local_crs_nc->fillComplete (A_local_->getDomainMap (), A_local_->getRangeMap ());
        A_local_crs = rcp_const_cast<const crs_matrix_type> (A_local_crs_nc);
      }
      Graph_ = rcp (new Ifpack2::IlukGraph<crs_graph_type,kk_handle_type> (A_local_crs->getCrsGraph (),
                                                                           LevelOfFill_, 0, Overalloc_));
    }

    if (this->isKokkosKernelsSpiluk_) Graph_->initialize (KernelHandle_);
    else Graph_->initialize ();

    allocate_L_and_U ();
    checkOrderingConsistency (*A_local_);
    L_solver_->setMatrix (L_);
    L_solver_->initialize ();
    L_solver_->compute ();//NOTE: It makes sense to do compute here because only the nonzero pattern is involved in trisolve compute
    U_solver_->setMatrix (U_);
    U_solver_->initialize ();
    U_solver_->compute ();//NOTE: It makes sense to do compute here because only the nonzero pattern is involved in trisolve compute

    // Do not call initAllValues. compute() always calls initAllValues to
    // fill L and U with possibly new numbers. initialize() is concerned
    // only with the nonzero pattern.
  } // Stop timing

  isInitialized_ = true;
  ++numInitialize_;
  initializeTime_ += (timer.wallTime() - startTime);
}

template<class MatrixType>
void
RILUK<MatrixType>::
checkOrderingConsistency (const row_matrix_type& A)
{
  // First check that the local row map ordering is the same as the local portion of the column map.
  // The extraction of the strictly lower/upper parts of A, as well as the factorization,
  // implicitly assume that this is the case.
  Teuchos::ArrayView<const global_ordinal_type> rowGIDs = A.getRowMap()->getLocalElementList();
  Teuchos::ArrayView<const global_ordinal_type> colGIDs = A.getColMap()->getLocalElementList();
  bool gidsAreConsistentlyOrdered=true;
  global_ordinal_type indexOfInconsistentGID=0;
  for (global_ordinal_type i=0; i<rowGIDs.size(); ++i) {
    if (rowGIDs[i] != colGIDs[i]) {
      gidsAreConsistentlyOrdered=false;
      indexOfInconsistentGID=i;
      break;
    }
  }
  TEUCHOS_TEST_FOR_EXCEPTION(gidsAreConsistentlyOrdered==false, std::runtime_error,
                             "The ordering of the local GIDs in the row and column maps is not the same"
                             << std::endl << "at index " << indexOfInconsistentGID
                             << ".  Consistency is required, as all calculations are done with"
                             << std::endl << "local indexing.");
}

template<class MatrixType>
void
RILUK<MatrixType>::
initAllValues (const row_matrix_type& A)
{
  using Teuchos::ArrayRCP;
  using Teuchos::Comm;
  using Teuchos::ptr;
  using Teuchos::RCP;
  using Teuchos::REDUCE_SUM;
  using Teuchos::reduceAll;
  typedef Tpetra::Map<local_ordinal_type,global_ordinal_type,node_type> map_type;

  size_t NumIn = 0, NumL = 0, NumU = 0;
  bool DiagFound = false;
  size_t NumNonzeroDiags = 0;
  size_t MaxNumEntries = A.getGlobalMaxNumRowEntries();

  // Allocate temporary space for extracting the strictly
  // lower and upper parts of the matrix A.
  nonconst_local_inds_host_view_type InI("InI",MaxNumEntries);
  Teuchos::Array<local_ordinal_type> LI(MaxNumEntries);
  Teuchos::Array<local_ordinal_type> UI(MaxNumEntries);
  nonconst_values_host_view_type InV("InV",MaxNumEntries);
  Teuchos::Array<scalar_type> LV(MaxNumEntries);
  Teuchos::Array<scalar_type> UV(MaxNumEntries);

  // Check if values should be inserted or replaced
  const bool ReplaceValues = L_->isStaticGraph () || L_->isLocallyIndexed ();

  L_->resumeFill ();
  U_->resumeFill ();
  if (ReplaceValues) {
    L_->setAllToScalar (STS::zero ()); // Zero out L and U matrices
    U_->setAllToScalar (STS::zero ());
  }

  D_->putScalar (STS::zero ()); // Set diagonal values to zero
  auto DV = Kokkos::subview(D_->getLocalViewHost(Tpetra::Access::ReadWrite), Kokkos::ALL(), 0);

  RCP<const map_type> rowMap = L_->getRowMap ();

  // First we copy the user's matrix into L and U, regardless of fill level.
  // It is important to note that L and U are populated using local indices.
  // This means that if the row map GIDs are not monotonically increasing
  // (i.e., permuted or gappy), then the strictly lower (upper) part of the
  // matrix is not the one that you would get if you based L (U) on GIDs.
  // This is ok, as the *order* of the GIDs in the rowmap is a better
  // expression of the user's intent than the GIDs themselves.

  Teuchos::ArrayView<const global_ordinal_type> nodeGIDs = rowMap->getLocalElementList();
  for (size_t myRow=0; myRow<A.getLocalNumRows(); ++myRow) {
    local_ordinal_type local_row = myRow;

    //TODO JJH 4April2014 An optimization is to use getLocalRowView.  Not all matrices support this,
    //                    we'd need to check via the Tpetra::RowMatrix method supportsRowViews().
    A.getLocalRowCopy (local_row, InI, InV, NumIn); // Get Values and Indices

    // Split into L and U (we don't assume that indices are ordered).

    NumL = 0;
    NumU = 0;
    DiagFound = false;

    for (size_t j = 0; j < NumIn; ++j) {
      const local_ordinal_type k = InI[j];

      if (k == local_row) {
        DiagFound = true;
        // Store perturbed diagonal in Tpetra::Vector D_
        DV(local_row) += Rthresh_ * InV[j] + IFPACK2_SGN(InV[j]) * Athresh_;
      }
      else if (k < 0) { // Out of range
        TEUCHOS_TEST_FOR_EXCEPTION(
          true, std::runtime_error, "Ifpack2::RILUK::initAllValues: current "
          "GID k = " << k << " < 0.  I'm not sure why this is an error; it is "
          "probably an artifact of the undocumented assumptions of the "
          "original implementation (likely copied and pasted from Ifpack).  "
          "Nevertheless, the code I found here insisted on this being an error "
          "state, so I will throw an exception here.");
      }
      else if (k < local_row) {
        LI[NumL] = k;
        LV[NumL] = InV[j];
        NumL++;
      }
      else if (Teuchos::as<size_t>(k) <= rowMap->getLocalNumElements()) {
        UI[NumU] = k;
        UV[NumU] = InV[j];
        NumU++;
      }
    }

    // Check in things for this row of L and U

    if (DiagFound) {
      ++NumNonzeroDiags;
    } else {
      DV(local_row) = Athresh_;
    }

    if (NumL) {
      if (ReplaceValues) {
        L_->replaceLocalValues(local_row, LI(0, NumL), LV(0,NumL));
      } else {
        //FIXME JJH 24April2014 Is this correct?  I believe this case is when there aren't already values
        //FIXME in this row in the column locations corresponding to UI.
        L_->insertLocalValues(local_row, LI(0,NumL), LV(0,NumL));
      }
    }

    if (NumU) {
      if (ReplaceValues) {
        U_->replaceLocalValues(local_row, UI(0,NumU), UV(0,NumU));
      } else {
        //FIXME JJH 24April2014 Is this correct?  I believe this case is when there aren't already values
        //FIXME in this row in the column locations corresponding to UI.
        U_->insertLocalValues(local_row, UI(0,NumU), UV(0,NumU));
      }
    }
  }

  // At this point L and U have the values of A in the structure of L
  // and U, and diagonal vector D

  isInitialized_ = true;
}


template<class MatrixType>
void RILUK<MatrixType>::compute ()
{
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::rcp_const_cast;
  using Teuchos::rcp_dynamic_cast;
  using Teuchos::Array;
  using Teuchos::ArrayView;
  const char prefix[] = "Ifpack2::RILUK::compute: ";

  // initialize() checks this too, but it's easier for users if the
  // error shows them the name of the method that they actually
  // called, rather than the name of some internally called method.
  TEUCHOS_TEST_FOR_EXCEPTION
    (A_.is_null (), std::runtime_error, prefix << "The matrix is null.  Please "
     "call setMatrix() with a nonnull input before calling this method.");
  TEUCHOS_TEST_FOR_EXCEPTION
    (! A_->isFillComplete (), std::runtime_error, prefix << "The matrix is not "
     "fill complete.  You may not invoke initialize() or compute() with this "
     "matrix until the matrix is fill complete.  If your matrix is a "
     "Tpetra::CrsMatrix, please call fillComplete on it (with the domain and "
     "range Maps, if appropriate) before calling this method.");

  if (! isInitialized ()) {
    initialize (); // Don't count this in the compute() time
  }

  Teuchos::Time timer ("RILUK::compute");

  // Start timing
  Teuchos::TimeMonitor timeMon (timer);
  double startTime = timer.wallTime();

  isComputed_ = false;

  if (!this->isKokkosKernelsSpiluk_) {
    // Fill L and U with numbers. This supports nonzero pattern reuse by calling
    // initialize() once and then compute() multiple times.
    initAllValues (*A_local_);

    // MinMachNum should be officially defined, for now pick something a little
    // bigger than IEEE underflow value

    const scalar_type MinDiagonalValue = STS::rmin ();
    const scalar_type MaxDiagonalValue = STS::one () / MinDiagonalValue;

    size_t NumIn, NumL, NumU;

    // Get Maximum Row length
    const size_t MaxNumEntries =
            L_->getLocalMaxNumRowEntries () + U_->getLocalMaxNumRowEntries () + 1;

    Teuchos::Array<local_ordinal_type> InI(MaxNumEntries); // Allocate temp space
    Teuchos::Array<scalar_type> InV(MaxNumEntries);
    size_t num_cols = U_->getColMap()->getLocalNumElements();
    Teuchos::Array<int> colflag(num_cols);

    auto DV = Kokkos::subview(D_->getLocalViewHost(Tpetra::Access::ReadWrite), Kokkos::ALL(), 0);

    // Now start the factorization.

    for (size_t j = 0; j < num_cols; ++j) {
      colflag[j] = -1;
    }
    using IST = typename row_matrix_type::impl_scalar_type;
    for (size_t i = 0; i < L_->getLocalNumRows (); ++i) {
      local_ordinal_type local_row = i;
      // Need some integer workspace and pointers
      size_t NumUU;
      local_inds_host_view_type UUI;
      values_host_view_type UUV;

      // Fill InV, InI with current row of L, D and U combined

      NumIn = MaxNumEntries;
      nonconst_local_inds_host_view_type InI_v(InI.data(),MaxNumEntries);
      nonconst_values_host_view_type     InV_v(reinterpret_cast<IST*>(InV.data()),MaxNumEntries);

      L_->getLocalRowCopy (local_row, InI_v , InV_v, NumL);

      InV[NumL] = DV(i); // Put in diagonal
      InI[NumL] = local_row;

      nonconst_local_inds_host_view_type InI_sub(InI.data()+NumL+1,MaxNumEntries-NumL-1);
      nonconst_values_host_view_type     InV_sub(reinterpret_cast<IST*>(InV.data())+NumL+1,MaxNumEntries-NumL-1);
  
      U_->getLocalRowCopy (local_row, InI_sub,InV_sub, NumU);
      NumIn = NumL+NumU+1;

      // Set column flags
      for (size_t j = 0; j < NumIn; ++j) {
        colflag[InI[j]] = j;
      }

      scalar_type diagmod = STS::zero (); // Off-diagonal accumulator

      for (size_t jj = 0; jj < NumL; ++jj) {
        local_ordinal_type j = InI[jj];
        IST multiplier = InV[jj]; // current_mults++;
        
        InV[jj] *= static_cast<scalar_type>(DV(j));
        
        U_->getLocalRowView(j, UUI, UUV); // View of row above
        NumUU = UUI.size();
        
        if (RelaxValue_ == STM::zero ()) {
          for (size_t k = 0; k < NumUU; ++k) {
            const int kk = colflag[UUI[k]];
            // FIXME (mfh 23 Dec 2013) Wait a second, we just set
            // colflag above using size_t (which is generally unsigned),
            // but now we're querying it using int (which is signed).
            if (kk > -1) {
              InV[kk] -= static_cast<scalar_type>(multiplier * UUV[k]);
            }
          }

        }
        else {
          for (size_t k = 0; k < NumUU; ++k) {
            // FIXME (mfh 23 Dec 2013) Wait a second, we just set
            // colflag above using size_t (which is generally unsigned),
            // but now we're querying it using int (which is signed).
            const int kk = colflag[UUI[k]];
            if (kk > -1) {
              InV[kk] -= static_cast<scalar_type>(multiplier*UUV[k]);
            }
            else {
              diagmod -= static_cast<scalar_type>(multiplier*UUV[k]);
            }
          }
        }
      }

      if (NumL) {
        // Replace current row of L
        L_->replaceLocalValues (local_row, InI (0, NumL), InV (0, NumL));
      }

      DV(i) = InV[NumL]; // Extract Diagonal value

      if (RelaxValue_ != STM::zero ()) {
        DV(i) += RelaxValue_*diagmod; // Add off diagonal modifications
      }

      if (STS::magnitude (DV(i)) > STS::magnitude (MaxDiagonalValue)) {
        if (STS::real (DV(i)) < STM::zero ()) {
          DV(i) = -MinDiagonalValue;
        }
        else {
          DV(i) = MinDiagonalValue;
        }
      }
      else {
        DV(i) = static_cast<impl_scalar_type>(STS::one ()) / DV(i); // Invert diagonal value
      }

      for (size_t j = 0; j < NumU; ++j) {
        InV[NumL+1+j] *= static_cast<scalar_type>(DV(i)); // Scale U by inverse of diagonal
      }

      if (NumU) {
        // Replace current row of L and U        
        U_->replaceLocalValues (local_row, InI (NumL+1, NumU), InV (NumL+1, NumU));
      }

      // Reset column flags
      for (size_t j = 0; j < NumIn; ++j) {
        colflag[InI[j]] = -1;
      }
    }

    // The domain of L and the range of U are exactly their own row maps
    // (there is no communication).  The domain of U and the range of L
    // must be the same as those of the original matrix, However if the
    // original matrix is a VbrMatrix, these two latter maps are
    // translation from a block map to a point map.
    // FIXME (mfh 23 Dec 2013) Do we know that the column Map of L_ is
    // always one-to-one?
    L_->fillComplete (L_->getColMap (), A_local_->getRangeMap ());
    U_->fillComplete (A_local_->getDomainMap (), U_->getRowMap ());

    // If L_solver_ or U_solver store modified factors internally, we need to reset those
    L_solver_->setMatrix (L_);
    L_solver_->compute ();//NOTE: Only do compute if the pointer changed. Otherwise, do nothing
    U_solver_->setMatrix (U_);
    U_solver_->compute ();//NOTE: Only do compute if the pointer changed. Otherwise, do nothing
  }
  else {
    {//Make sure values in A is picked up even in case of pattern reuse
      RCP<const crs_matrix_type> A_local_crs = Details::getCrsMatrix(A_local_);
      if(A_local_crs.is_null()) {
        local_ordinal_type numRows = A_local_->getLocalNumRows();
        Array<size_t> entriesPerRow(numRows);
        for(local_ordinal_type i = 0; i < numRows; i++) {
          entriesPerRow[i] = A_local_->getNumEntriesInLocalRow(i);
        }
        RCP<crs_matrix_type> A_local_crs_nc =
          rcp (new crs_matrix_type (A_local_->getRowMap (),
                                    A_local_->getColMap (),
                                    entriesPerRow()));
        // copy entries into A_local_crs
        nonconst_local_inds_host_view_type indices("indices",A_local_->getLocalMaxNumRowEntries());
        nonconst_values_host_view_type values("values",A_local_->getLocalMaxNumRowEntries());
        for(local_ordinal_type i = 0; i < numRows; i++) {
          size_t numEntries = 0;
          A_local_->getLocalRowCopy(i, indices, values, numEntries);
          A_local_crs_nc->insertLocalValues(i, numEntries, reinterpret_cast<scalar_type*>(values.data()),indices.data());
        }
        A_local_crs_nc->fillComplete (A_local_->getDomainMap (), A_local_->getRangeMap ());
        A_local_crs = rcp_const_cast<const crs_matrix_type> (A_local_crs_nc);
      }
      auto lclMtx = A_local_crs->getLocalMatrixDevice();
      A_local_rowmap_  = lclMtx.graph.row_map;
      A_local_entries_ = lclMtx.graph.entries;
      A_local_values_  = lclMtx.values;
    }

    L_->resumeFill ();
    U_->resumeFill ();

    if (L_->isStaticGraph () || L_->isLocallyIndexed ()) {
      L_->setAllToScalar (STS::zero ()); // Zero out L and U matrices
      U_->setAllToScalar (STS::zero ());
    }
    
    using row_map_type = typename crs_matrix_type::local_matrix_device_type::row_map_type;

    {
      auto lclL = L_->getLocalMatrixDevice();
      row_map_type L_rowmap  = lclL.graph.row_map;
      auto L_entries = lclL.graph.entries;
      auto L_values  = lclL.values;

      auto lclU = U_->getLocalMatrixDevice();
      row_map_type U_rowmap  = lclU.graph.row_map;
      auto U_entries = lclU.graph.entries;
      auto U_values  = lclU.values;

      KokkosSparse::Experimental::spiluk_numeric( KernelHandle_.getRawPtr(), LevelOfFill_, 
                                                  A_local_rowmap_, A_local_entries_, A_local_values_, 
                                                  L_rowmap, L_entries, L_values, U_rowmap, U_entries, U_values );
    }
    
    L_->fillComplete (L_->getColMap (), A_local_->getRangeMap ());
    U_->fillComplete (A_local_->getDomainMap (), U_->getRowMap ());
    
    L_solver_->setMatrix (L_);
    L_solver_->compute ();//NOTE: Only do compute if the pointer changed. Otherwise, do nothing
    U_solver_->setMatrix (U_);
    U_solver_->compute ();//NOTE: Only do compute if the pointer changed. Otherwise, do nothing
  }

  isComputed_ = true;
  ++numCompute_;
  computeTime_ += (timer.wallTime() - startTime);
}


template<class MatrixType>
void
RILUK<MatrixType>::
apply (const Tpetra::MultiVector<scalar_type,local_ordinal_type,global_ordinal_type,node_type>& X,
       Tpetra::MultiVector<scalar_type,local_ordinal_type,global_ordinal_type,node_type>& Y,
       Teuchos::ETransp mode,
       scalar_type alpha,
       scalar_type beta) const
{
  using Teuchos::RCP;
  using Teuchos::rcpFromRef;

  TEUCHOS_TEST_FOR_EXCEPTION(
    A_.is_null (), std::runtime_error, "Ifpack2::RILUK::apply: The matrix is "
    "null.  Please call setMatrix() with a nonnull input, then initialize() "
    "and compute(), before calling this method.");
  TEUCHOS_TEST_FOR_EXCEPTION(
    ! isComputed (), std::runtime_error,
    "Ifpack2::RILUK::apply: If you have not yet called compute(), "
    "you must call compute() before calling this method.");
  TEUCHOS_TEST_FOR_EXCEPTION(
    X.getNumVectors () != Y.getNumVectors (), std::invalid_argument,
    "Ifpack2::RILUK::apply: X and Y do not have the same number of columns.  "
    "X.getNumVectors() = " << X.getNumVectors ()
    << " != Y.getNumVectors() = " << Y.getNumVectors () << ".");
  TEUCHOS_TEST_FOR_EXCEPTION(
    STS::isComplex && mode == Teuchos::CONJ_TRANS, std::logic_error,
    "Ifpack2::RILUK::apply: mode = Teuchos::CONJ_TRANS is not implemented for "
    "complex Scalar type.  Please talk to the Ifpack2 developers to get this "
    "fixed.  There is a FIXME in this file about this very issue.");
#ifdef HAVE_IFPACK2_DEBUG
  {
    const magnitude_type D_nrm1 = D_->norm1 ();
    TEUCHOS_TEST_FOR_EXCEPTION( STM::isnaninf (D_nrm1), std::runtime_error, "Ifpack2::RILUK::apply: The 1-norm of the stored diagonal is NaN or Inf.");
    Teuchos::Array<magnitude_type> norms (X.getNumVectors ());
    X.norm1 (norms ());
    bool good = true;
    for (size_t j = 0; j < X.getNumVectors (); ++j) {
      if (STM::isnaninf (norms[j])) {
        good = false;
        break;
      }
    }
    TEUCHOS_TEST_FOR_EXCEPTION( ! good, std::runtime_error, "Ifpack2::RILUK::apply: The 1-norm of the input X is NaN or Inf.");
  }
#endif // HAVE_IFPACK2_DEBUG

  const scalar_type one = STS::one ();
  const scalar_type zero = STS::zero ();

  Teuchos::Time timer ("RILUK::apply");
  double startTime = timer.wallTime();
  { // Start timing
    Teuchos::TimeMonitor timeMon (timer);
    if (alpha == one && beta == zero) {
      if (mode == Teuchos::NO_TRANS) { // Solve L (D (U Y)) = X for Y.      
        // Start by solving L Y = X for Y.
        L_solver_->apply (X, Y, mode);

        if (!this->isKokkosKernelsSpiluk_) {
          // Solve D Y = Y.  The operation lets us do this in place in Y, so we can
          // write "solve D Y = Y for Y."
          Y.elementWiseMultiply (one, *D_, Y, zero);
        }

        U_solver_->apply (Y, Y, mode); // Solve U Y = Y.
      }
      else { // Solve U^P (D^P (L^P Y)) = X for Y (where P is * or T).      
        // Start by solving U^P Y = X for Y.
        U_solver_->apply (X, Y, mode);

        if (!this->isKokkosKernelsSpiluk_) {
          // Solve D^P Y = Y.
          //
          // FIXME (mfh 24 Jan 2014) If mode = Teuchos::CONJ_TRANS, we
          // need to do an elementwise multiply with the conjugate of
          // D_, not just with D_ itself.
          Y.elementWiseMultiply (one, *D_, Y, zero);
	    }

        L_solver_->apply (Y, Y, mode); // Solve L^P Y = Y.
      }
    }
    else { // alpha != 1 or beta != 0
      if (alpha == zero) {
        // The special case for beta == 0 ensures that if Y contains Inf
        // or NaN values, we replace them with 0 (following BLAS
        // convention), rather than multiplying them by 0 to get NaN.
        if (beta == zero) {
          Y.putScalar (zero);
        } else {
          Y.scale (beta);
        }
      } else { // alpha != zero
        MV Y_tmp (Y.getMap (), Y.getNumVectors ());
        apply (X, Y_tmp, mode);
        Y.update (alpha, Y_tmp, beta);
      }
    }
  }//end timing

#ifdef HAVE_IFPACK2_DEBUG
  {
    Teuchos::Array<magnitude_type> norms (Y.getNumVectors ());
    Y.norm1 (norms ());
    bool good = true;
    for (size_t j = 0; j < Y.getNumVectors (); ++j) {
      if (STM::isnaninf (norms[j])) {
        good = false;
        break;
      }
    }
    TEUCHOS_TEST_FOR_EXCEPTION( ! good, std::runtime_error, "Ifpack2::RILUK::apply: The 1-norm of the output Y is NaN or Inf.");
  }
#endif // HAVE_IFPACK2_DEBUG

  ++numApply_;
  applyTime_ += (timer.wallTime() - startTime);
}


//VINH comment out since multiply() is not needed anywhere
//template<class MatrixType>
//void RILUK<MatrixType>::
//multiply (const Tpetra::MultiVector<scalar_type,local_ordinal_type,global_ordinal_type,node_type>& X,
//          Tpetra::MultiVector<scalar_type,local_ordinal_type,global_ordinal_type,node_type>& Y,
//          const Teuchos::ETransp mode) const
//{
//  const scalar_type zero = STS::zero ();
//  const scalar_type one = STS::one ();
//
//  if (mode != Teuchos::NO_TRANS) {
//    U_->apply (X, Y, mode); //
//    Y.update (one, X, one); // Y = Y + X (account for implicit unit diagonal)
//
//    // FIXME (mfh 24 Jan 2014) If mode = Teuchos::CONJ_TRANS, we need
//    // to do an elementwise multiply with the conjugate of D_, not
//    // just with D_ itself.
//    Y.elementWiseMultiply (one, *D_, Y, zero); // y = D*y (D_ has inverse of diagonal)
//
//    MV Y_tmp (Y, Teuchos::Copy); // Need a temp copy of Y
//    L_->apply (Y_tmp, Y, mode);
//    Y.update (one, Y_tmp, one); // (account for implicit unit diagonal)
//  }
//  else {
//    L_->apply (X, Y, mode);
//    Y.update (one, X, one); // Y = Y + X (account for implicit unit diagonal)
//    Y.elementWiseMultiply (one, *D_, Y, zero); // y = D*y (D_ has inverse of diagonal)
//    MV Y_tmp (Y, Teuchos::Copy); // Need a temp copy of Y1
//    U_->apply (Y_tmp, Y, mode);
//    Y.update (one, Y_tmp, one); // (account for implicit unit diagonal)
//  }
//}

template<class MatrixType>
std::string RILUK<MatrixType>::description () const
{
  std::ostringstream os;

  // Output is a valid YAML dictionary in flow style.  If you don't
  // like everything on a single line, you should call describe()
  // instead.
  os << "\"Ifpack2::RILUK\": {";
  os << "Initialized: " << (isInitialized () ? "true" : "false") << ", "
     << "Computed: " << (isComputed () ? "true" : "false") << ", ";

  os << "Level-of-fill: " << getLevelOfFill() << ", ";

  if (A_.is_null ()) {
    os << "Matrix: null";
  }
  else {
    os << "Global matrix dimensions: ["
       << A_->getGlobalNumRows () << ", " << A_->getGlobalNumCols () << "]"
       << ", Global nnz: " << A_->getGlobalNumEntries();
  }

  if (! L_solver_.is_null ()) os << ", " << L_solver_->description ();
  if (! U_solver_.is_null ()) os << ", " << U_solver_->description ();

  os << "}";
  return os.str ();
}

} // namespace Ifpack2

#define IFPACK2_RILUK_INSTANT(S,LO,GO,N)                            \
  template class Ifpack2::RILUK< Tpetra::RowMatrix<S, LO, GO, N> >;

#endif
