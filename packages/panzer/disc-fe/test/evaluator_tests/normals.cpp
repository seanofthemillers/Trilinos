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
#include "Panzer_CellData.hpp"
#include "Panzer_LocalMeshInfo.hpp"
#include "Panzer_Workset.hpp"
#include "Panzer_Traits.hpp"
#include "Panzer_CommonArrayFactories.hpp"

#include "Panzer_Normals.hpp"

#include "Phalanx_FieldManager.hpp"

// for making explicit instantiated tests easier
#define UNIT_TEST_GROUP(TYPE) \
  TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT(normals,test2d,TYPE)

namespace panzer {

TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL(normals,test2d,EvalType)
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
   Kokkos::parallel_for(1, KOKKOS_LAMBDA (int) {
         coords(0,0,0) = 1.0; coords(0,0,1) = 0.0;
         coords(0,1,0) = 1.0; coords(0,1,1) = 1.0;
         coords(0,2,0) = 0.0; coords(0,2,1) = 1.0;
         coords(0,3,0) = 0.0; coords(0,3,1) = 0.0;

         coords(1,0,0) = 1.0; coords(1,0,1) = 1.0;
         coords(1,1,0) = 2.0; coords(1,1,1) = 2.0;
         coords(1,2,0) = 1.0; coords(1,2,1) = 3.0;
         coords(1,3,0) = 0.0; coords(1,3,1) = 2.0;
      });

   panzer::WorksetOptions options;
   options.side_assembly_ = true;
      
   worksets->push_back(panzer::Workset());
   worksets->back().setup(partition, options);
  }


  int quadOrder = 5;
  panzer::CellData cellData(numCells,side,topo);
  Teuchos::RCP<panzer::IntegrationRule> quadRule = Teuchos::rcp(new panzer::IntegrationRule(quadOrder,cellData));
  out << "num quad points = " << quadRule->num_points << std::endl;

  Teuchos::RCP<PHX::FieldManager<panzer::Traits> > fm
     = Teuchos::rcp(new PHX::FieldManager<panzer::Traits>);

  // typedef panzer::Traits::Residual EvalType;
  Teuchos::RCP<PHX::MDField<typename EvalType::ScalarT,panzer::Cell,panzer::Point,panzer::Dim> > normalsPtr;
  {
     Teuchos::ParameterList p;
     p.set("Name","Norms");
     p.set("IR",quadRule);
     p.set("Side ID",side);

     RCP<panzer::Normals<EvalType,panzer::Traits> > normEval
        = rcp(new panzer::Normals<EvalType,panzer::Traits>(p));
     RCP<PHX::Evaluator<panzer::Traits> > eval = normEval;

     fm->registerEvaluator<EvalType>(eval);
     fm->requireField<EvalType>(normEval->getFieldTag());

     const PHX::FieldTag & ft = normEval->getFieldTag();
     normalsPtr = rcp(new
         PHX::MDField<typename EvalType::ScalarT,panzer::Cell,panzer::Point,panzer::Dim>(ft.name(),quadRule->dl_vector));
  }
  PHX::MDField<typename EvalType::ScalarT,panzer::Cell,panzer::Point,panzer::Dim> & normals = *normalsPtr;

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

  fm->getFieldData<EvalType>(normals);

  TEST_EQUALITY(normals.rank(),3);
  TEST_EQUALITY(static_cast<int>(normals.size()),numCells*quadRule->num_points*dim);
  auto normals_v = normals.get_static_view();
  auto normals_h = Kokkos::create_mirror_view ( normals_v);
  Kokkos::deep_copy(normals_h, normals_v);

  normals.print(out,false);
  for(int i=0;i<numCells;i++) {

     // useful for checking if normals are consistent: transformation is
     // affine!
     double nx0 = Sacado::scalarValue(normals_h(i,0,0));
     double ny0 = Sacado::scalarValue(normals_h(i,0,1));

     for(int v=0;v<quadRule->num_points;v++) {
        double nx = Sacado::scalarValue(normals_h(i,v,0));
        double ny = Sacado::scalarValue(normals_h(i,v,1));

        TEST_FLOATING_EQUALITY(nx*nx+ny*ny,1.0,1e-15);

        // check point consistency
        TEST_FLOATING_EQUALITY(nx,nx0,1e-15);
        TEST_FLOATING_EQUALITY(ny,ny0,1e-15);
     }
  }

  // check cell 0
  {
     double nx = Sacado::scalarValue(normals_h(0,0,0));
     double ny = Sacado::scalarValue(normals_h(0,0,1));

     TEST_FLOATING_EQUALITY(nx,0.0,1e-15);
     TEST_FLOATING_EQUALITY(ny,1.0,1e-15);
  }

  // check cell 1
  {
     double nx = Sacado::scalarValue(normals_h(1,0,0));
     double ny = Sacado::scalarValue(normals_h(1,0,1));
     double sqrt2 = std::sqrt(2.0);

     TEST_FLOATING_EQUALITY(nx,1.0/sqrt2,1e-15);
     TEST_FLOATING_EQUALITY(ny,1.0/sqrt2,1e-15);
  }
}

typedef Traits::Residual ResidualType;
typedef Traits::Jacobian JacobianType;

UNIT_TEST_GROUP(ResidualType)
UNIT_TEST_GROUP(JacobianType)

#ifdef Panzer_BUILD_HESSIAN_SUPPORT
typedef Traits::Hessian HessianType;
UNIT_TEST_GROUP(HessianType)
#endif

}
