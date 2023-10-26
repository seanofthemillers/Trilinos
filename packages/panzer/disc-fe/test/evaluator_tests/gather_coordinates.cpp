// @HEADER
// ***********************************************************************
//
//           Panzer: A partial differential equation assembly
//       engine for strongly coupled complex multiphysics systems
//                 Copyright (2011) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
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
// Questions? Contact Roger P. Pawlowski (rppawlo@sandia.gov) and
// Eric C. Cyr (eccyr@sandia.gov)
// ***********************************************************************
// @HEADER

#include <Teuchos_ConfigDefs.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_TimeMonitor.hpp>

using Teuchos::RCP;
using Teuchos::rcp;

#include "Teuchos_DefaultComm.hpp"
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_DefaultMpiComm.hpp"
#include "Teuchos_OpaqueWrapper.hpp"

#include "Kokkos_View_Fad.hpp"
#include "PanzerDiscFE_config.hpp"
#include "Panzer_IntegrationRule.hpp"
#include "Panzer_IntegrationValues2.hpp"
#include "Panzer_BasisValues2.hpp"
#include "Panzer_CellData.hpp"
#include "Panzer_LocalMeshInfo.hpp"
#include "Panzer_Workset.hpp"
#include "Panzer_Traits.hpp"
#include "Panzer_CommonArrayFactories.hpp"

#include "Panzer_GatherBasisCoordinates.hpp"
#include "Panzer_GatherIntegrationCoordinates.hpp"

#include "Phalanx_FieldManager.hpp"

#include "UnitValueEvaluator.hpp"

// for making explicit instantiated tests easier
#define UNIT_TEST_GROUP(TYPE) \
  TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT(gather_coordinates,basis,TYPE) \
  TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT(gather_coordinates,integration,TYPE)

namespace panzer {

TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL(gather_coordinates,basis,EvalType)
{

  // build global (or serial communicator)
  #ifdef HAVE_MPI
     Teuchos::RCP<const Teuchos::MpiComm<int> > eComm = Teuchos::rcp(new Teuchos::MpiComm<int>(MPI_COMM_WORLD));
  #else
      auto eComm = Teuchos::rcp(Teuchos::DefaultComm<int>::getComm());
  #endif

  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::rcp_dynamic_cast;

  // panzer::pauseToAttach();

  // build a dummy workset
  //////////////////////////////////////////////////////////
  // typedef Kokkos::DynRankView<double,PHX::Device> FieldArray;
  int numCells = 2, numVerts = 4, dim = 2, side=1;
  auto topo = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData< shards::Quadrilateral<4> >()));
  auto worksets = Teuchos::rcp(new std::vector<panzer::Workset>());
  {
    panzer::LocalMeshPartition partition;
    partition.num_owned_cells = numCells;
    partition.num_ghstd_cells = partition.num_virtual_cells = 0;
    partition.subcell_dimension = dim-1;
    partition.subcell_index = side;
    partition.local_cells = PHX::View<panzer::LocalOrdinal*>("local_cells",numCells);
    partition.global_cells = PHX::View<panzer::GlobalOrdinal*>("global_cells",numCells);
    partition.cell_nodes = PHX::View<double***>("cell_nodes",numCells,numVerts,dim);
    partition.cell_topology = topo;

    auto & coords = partition.cell_nodes;
    Kokkos::parallel_for(1, KOKKOS_LAMBDA (int ) {
        coords(0,0,0) = 1.0; coords(0,0,1) = 0.0;
        coords(0,1,0) = 1.0; coords(0,1,1) = 1.0;
        coords(0,2,0) = 0.0; coords(0,2,1) = 1.0;
        coords(0,3,0) = 0.0; coords(0,3,1) = 0.0;

        coords(1,0,0) = 1.0; coords(1,0,1) = 1.0;
        coords(1,1,0) = 2.0; coords(1,1,1) = 2.0;
        coords(1,2,0) = 1.0; coords(1,2,1) = 3.0;
        coords(1,3,0) = 0.0; coords(1,3,1) = 2.0;
      });
    Kokkos::fence();
    panzer::WorksetOptions options;
    options.side_assembly_ = true;
      
    worksets->push_back(panzer::Workset());
    worksets->back().setup(partition, options);
  }

  auto coords = (*worksets)[0].getCellNodes();

  // build topology, basis, integration rule, and basis layout
  int quadOrder = 5;
  panzer::CellData cellData(numCells,side,topo);
  RCP<PureBasis> basis = rcp(new PureBasis("Q1",1,2,topo));
  Teuchos::RCP<panzer::IntegrationRule> quadRule = Teuchos::rcp(new panzer::IntegrationRule(quadOrder,cellData));
  RCP<BasisIRLayout> basisLayout = rcp(new BasisIRLayout(basis,*quadRule));

  RCP<IntegrationValues2<double> > quadValues = rcp(new IntegrationValues2<double>("",true));
  quadValues->setupArrays(quadRule);
  quadValues->evaluateValues(coords);

  Teuchos::RCP<PHX::FieldManager<panzer::Traits> > fm
     = Teuchos::rcp(new PHX::FieldManager<panzer::Traits>);

  // typedef panzer::Traits::Residual EvalType;
  Teuchos::RCP<PHX::MDField<typename EvalType::ScalarT,panzer::Cell,panzer::BASIS,panzer::Dim> > fmCoordsPtr;
  Teuchos::RCP<PHX::DataLayout> dl_coords = basis->coordinates;

  // add in some evaluators
  ///////////////////////////////////////////////////
  {
     RCP<panzer::GatherBasisCoordinates<EvalType,panzer::Traits> > eval
        = rcp(new panzer::GatherBasisCoordinates<EvalType,panzer::Traits>(*basis));

     const std::vector<RCP<PHX::FieldTag> > & evalFields = eval->evaluatedFields();
     const std::vector<RCP<PHX::FieldTag> > & depFields = eval->dependentFields();

     TEST_EQUALITY(evalFields.size(),1);
     TEST_EQUALITY(depFields.size(), 0);
     std::string ref_fieldName = GatherBasisCoordinates<EvalType,panzer::Traits>::fieldName(basis->name());
     TEST_EQUALITY(ref_fieldName,evalFields[0]->name());

     fm->registerEvaluator<EvalType>(eval);

     const PHX::FieldTag & ft = *evalFields[0];
     fm->requireField<EvalType>(ft);
     fmCoordsPtr = rcp(new
         PHX::MDField<typename EvalType::ScalarT,panzer::Cell,panzer::BASIS,panzer::Dim>(ft.name(),dl_coords));
  }

  PHX::MDField<typename EvalType::ScalarT,panzer::Cell,panzer::BASIS,panzer::Dim> & fmCoords = *fmCoordsPtr;

  panzer::Traits::SD setupData;
  setupData.worksets_ = worksets;

  std::vector<PHX::index_size_type> derivative_dimensions;
  derivative_dimensions.push_back(4);
  fm->setKokkosExtendedDataTypeDimensions<panzer::Traits::Jacobian>(derivative_dimensions);
#ifdef Panzer_BUILD_HESSIAN_SUPPORT
  fm->setKokkosExtendedDataTypeDimensions<panzer::Traits::Hessian>(derivative_dimensions);
#endif
  fm->postRegistrationSetup(setupData);

  panzer::Traits::PED preEvalData;

  fm->preEvaluate<EvalType>(preEvalData);
  fm->evaluateFields<EvalType>((*worksets)[0]);
  fm->postEvaluate<EvalType>(0);

  fm->getFieldData<EvalType>(fmCoords);

  fmCoords.print(out,true);
  int error = false;
  Kokkos::parallel_reduce(fmCoords.extent_int(0), KOKKOS_LAMBDA (int cell, int &err) {
    for(int pt=0;pt<fmCoords.extent_int(1);++pt)
      for(int d=0;d<fmCoords.extent_int(2);++d)
	err |= Sacado::scalarValue(fmCoords(cell,pt,d))!=coords(cell,pt,d);
    },error);
  TEST_EQUALITY(error, false);
}

TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL(gather_coordinates,integration,EvalType)
{

  // build global (or serial communicator)
  #ifdef HAVE_MPI
     Teuchos::RCP<const Teuchos::MpiComm<int> > eComm = Teuchos::rcp(new Teuchos::MpiComm<int>(MPI_COMM_WORLD));
  #else
      auto eComm = Teuchos::rcp(Teuchos::DefaultComm<int>::getComm());
  #endif

  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::rcp_dynamic_cast;

  // panzer::pauseToAttach();

  // build a dummy workset
  //////////////////////////////////////////////////////////
  // typedef Kokkos::DynRankView<double,PHX::Device> FieldArray;
  int numCells = 2, numVerts = 4, dim = 2, side=1;
  auto topo = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData< shards::Quadrilateral<4> >()));
  auto worksets = Teuchos::rcp(new std::vector<panzer::Workset>());
  {
    panzer::LocalMeshPartition partition;
    partition.num_owned_cells = numCells;
    partition.num_ghstd_cells = partition.num_virtual_cells = 0;
    partition.subcell_dimension = dim-1;
    partition.subcell_index = side;
    partition.local_cells = PHX::View<panzer::LocalOrdinal*>("local_cells",numCells);
    partition.global_cells = PHX::View<panzer::GlobalOrdinal*>("global_cells",numCells);
    partition.cell_nodes = PHX::View<double***>("cell_nodes",numCells,numVerts,dim);
    partition.cell_topology = topo;

    auto & coords = partition.cell_nodes;
    Kokkos::parallel_for(1, KOKKOS_LAMBDA (int ) {
        coords(0,0,0) = 1.0; coords(0,0,1) = 0.0;
        coords(0,1,0) = 1.0; coords(0,1,1) = 1.0;
        coords(0,2,0) = 0.0; coords(0,2,1) = 1.0;
        coords(0,3,0) = 0.0; coords(0,3,1) = 0.0;

        coords(1,0,0) = 1.0; coords(1,0,1) = 1.0;
        coords(1,1,0) = 2.0; coords(1,1,1) = 2.0;
        coords(1,2,0) = 1.0; coords(1,2,1) = 3.0;
        coords(1,3,0) = 0.0; coords(1,3,1) = 2.0;
      });
    Kokkos::fence();
    panzer::WorksetOptions options;
    options.side_assembly_ = true;
      
    worksets->push_back(panzer::Workset());
    worksets->back().setup(partition, options);
  }

  auto coords = (*worksets)[0].getCellNodes();

  // build topology, basis, integration rule, and basis layout
  int quadOrder = 5;
  panzer::CellData cellData(numCells,side,topo);
  RCP<PureBasis> basis = rcp(new PureBasis("Q1",1,2,topo));
  Teuchos::RCP<panzer::IntegrationRule> quadRule = Teuchos::rcp(new panzer::IntegrationRule(quadOrder,cellData));
  RCP<BasisIRLayout> basisLayout = rcp(new BasisIRLayout(basis,*quadRule));

  RCP<IntegrationValues2<double> > quadValues = rcp(new IntegrationValues2<double>("",true));
  quadValues->setupArrays(quadRule);
  quadValues->evaluateValues(coords);

  Teuchos::RCP<PHX::FieldManager<panzer::Traits> > fm
     = Teuchos::rcp(new PHX::FieldManager<panzer::Traits>);

  // typedef panzer::Traits::Residual EvalType;
  Teuchos::RCP<PHX::MDField<typename EvalType::ScalarT,panzer::Cell,panzer::Point,panzer::Dim> > fmCoordsPtr;
  Teuchos::RCP<PHX::DataLayout> dl_coords = quadRule->dl_vector;

  // add in some evaluators
  ///////////////////////////////////////////////////
  {
     RCP<panzer::GatherIntegrationCoordinates<EvalType,panzer::Traits> > eval
        = rcp(new panzer::GatherIntegrationCoordinates<EvalType,panzer::Traits>(*quadRule));

     const std::vector<RCP<PHX::FieldTag> > & evalFields = eval->evaluatedFields();
     const std::vector<RCP<PHX::FieldTag> > & depFields = eval->dependentFields();

     TEST_EQUALITY(evalFields.size(),1);
     TEST_EQUALITY(depFields.size(), 0);
     std::string ref_fieldName = GatherIntegrationCoordinates<EvalType,panzer::Traits>::fieldName(quadRule->cubature_degree);
     TEST_EQUALITY(ref_fieldName,evalFields[0]->name());

     fm->registerEvaluator<EvalType>(eval);

     const PHX::FieldTag & ft = *evalFields[0];
     fm->requireField<EvalType>(ft);
     fmCoordsPtr = rcp(new
         PHX::MDField<typename EvalType::ScalarT,panzer::Cell,panzer::Point,panzer::Dim>(ft.name(),dl_coords));
  }

  PHX::MDField<typename EvalType::ScalarT,panzer::Cell,panzer::Point,panzer::Dim> & fmCoords = *fmCoordsPtr;

  panzer::Traits::SD setupData;
  setupData.worksets_ = worksets;

  std::vector<PHX::index_size_type> derivative_dimensions;
  derivative_dimensions.push_back(4);
  fm->setKokkosExtendedDataTypeDimensions<panzer::Traits::Jacobian>(derivative_dimensions);
#ifdef Panzer_BUILD_HESSIAN_SUPPORT
  fm->setKokkosExtendedDataTypeDimensions<panzer::Traits::Hessian>(derivative_dimensions);
#endif
  fm->postRegistrationSetup(setupData);

  panzer::Traits::PED preEvalData;

  fm->preEvaluate<EvalType>(preEvalData);
  fm->evaluateFields<EvalType>((*worksets)[0]);
  fm->postEvaluate<EvalType>(0);

  fm->getFieldData<EvalType>(fmCoords);

  fmCoords.print(out,true);

  int error = false;
  auto q_coords = quadValues->ip_coordinates;
  Kokkos::parallel_reduce(fmCoords.extent_int(0), KOKKOS_LAMBDA (int cell, int &err) {
    for(int pt=0;pt<fmCoords.extent_int(1);++pt)
      for(int d=0;d<fmCoords.extent_int(2);++d)
	err |= Sacado::scalarValue(fmCoords(cell,pt,d)) != q_coords(cell,pt,d);
    },error);
  TEST_EQUALITY(error, false);

}

////////////////////////////////////////////////////////////////////////////////////
// Explicit instantiation
////////////////////////////////////////////////////////////////////////////////////

typedef Traits::Residual ResidualType;
typedef Traits::Jacobian JacobianType;

UNIT_TEST_GROUP(ResidualType)
UNIT_TEST_GROUP(JacobianType)

#ifdef Panzer_BUILD_HESSIAN_SUPPORT
typedef Traits::Hessian HessianType;
UNIT_TEST_GROUP(HessianType)
#endif

}
