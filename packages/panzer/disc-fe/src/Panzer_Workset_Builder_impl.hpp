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

#ifndef __Panzer_Workset_Builder_impl_hpp__
#define __Panzer_Workset_Builder_impl_hpp__

#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include "Panzer_LocalMeshInfo.hpp"
#include "Panzer_Workset.hpp"
#include "Panzer_OrientationsInterface.hpp"
#include "Panzer_CellData.hpp"
#include "Panzer_BC.hpp"
#include "Panzer_Shards_Utilities.hpp"
#include "Panzer_CommonArrayFactories.hpp"

#include "Phalanx_DataLayout_MDALayout.hpp"

// Intrepid2
#include "Shards_CellTopology.hpp"
#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"
#include "Intrepid2_Basis.hpp"

namespace
{

void
setupSimpleWorkset(const std::string & element_block,
                   const std::string & sideset,
                   const bool side_assembly,
                   const int subcell_dimension,
                   const int subcell_index,
                   const int num_cells,
                   Teuchos::RCP<const shards::CellTopology> cell_topology,
                   PHX::View<const panzer::LocalOrdinal*> local_cell_ids,
                   PHX::View<const double***> node_coordinates,
                   Teuchos::RCP<const panzer::OrientationsInterface> orientations,
                   panzer::Workset & workset)
{

  const size_t num_verts = node_coordinates.extent(1);
  const size_t num_dims = cell_topology->getDimension();
  const size_t num_coord_dims = node_coordinates.extent(2);
  
  panzer::LocalMeshPartition partition;
  partition.num_owned_cells = num_cells;
  partition.num_ghstd_cells = partition.num_virtual_cells = 0;
  partition.element_block_name = element_block;
  partition.sideset_name = sideset;
  partition.subcell_dimension = subcell_dimension;
  partition.subcell_index = subcell_index;
  partition.local_cells = PHX::View<panzer::LocalOrdinal*>("local_cells",num_cells);
  partition.global_cells = PHX::View<panzer::GlobalOrdinal*>("global_cells",num_cells);
  partition.cell_nodes = PHX::View<double***>("cell_nodes",num_cells,num_verts,num_dims);
  partition.cell_topology = cell_topology;
  partition.has_connectivity = false;

  TEUCHOS_ASSERT(num_coord_dims <= num_dims);
  // Currently no way to access this, so we fill with zeros
  // TODO: Do we even need to allocate this if we are not sub-partitioning?
  Kokkos::deep_copy(partition.global_cells, 0);

  // Fill coo
  auto & coords = partition.cell_nodes;
  if(num_coord_dims != num_dims)
    Kokkos::deep_copy(coords, 0);
  Kokkos::parallel_for(num_cells,KOKKOS_LAMBDA (int i){
    for(int j=0; j<num_verts; ++j)
      for(int k=0; k<num_coord_dims; ++k)
        coords(i,j,k) = node_coordinates(i,j,k);
  });

  auto & local_ids = partition.local_cells;
  TEUCHOS_ASSERT(local_cell_ids.extent(0) >= local_ids.extent(0));
  Kokkos::parallel_for(num_cells,KOKKOS_LAMBDA (int i){
    local_ids(i) = local_cell_ids(i);
  });
  Kokkos::fence();

  panzer::WorksetOptions options;
  options.side_assembly_ = side_assembly;
  options.orientations_ = orientations;
  workset.setup(partition,options);  
}

}

template<typename ArrayT>
Teuchos::RCP< std::vector<panzer::Workset> >
panzer::buildWorksets(const WorksetNeeds & needs,
                      const std::string & elementBlock,
		                  const std::vector<std::size_t>& local_cell_ids,
		                  const ArrayT& node_coordinates,
                      Teuchos::RCP<const panzer::OrientationsInterface> orientations)
{
  auto topo = needs.cellData.getCellTopology();
  TEUCHOS_ASSERT(topo != Teuchos::null);
  return buildSubcellWorksets(needs,elementBlock,false,topo->getDimension(),0,local_cell_ids,node_coordinates,orientations);
}

template<typename ArrayT>
Teuchos::RCP<std::vector<panzer::Workset> > 
panzer::buildSubcellWorksets(const WorksetNeeds & needs,
                             const std::string & elementBlock,
                             const bool side_assembly,
                             const int subcell_dim,
                             const int subcell_index,
                             const std::vector<std::size_t>& local_cell_ids,
                             const ArrayT& node_coordinates,
                             Teuchos::RCP<const panzer::OrientationsInterface> orientations)
{
  using std::vector;
  using std::string;
  using Teuchos::RCP;
  using Teuchos::rcp;

  panzer::MDFieldArrayFactory mdArrayFactory("",true);

  std::size_t total_num_cells = local_cell_ids.size();

  std::size_t workset_size = needs.cellData.numCells();
  auto cell_topology = needs.cellData.getCellTopology();

  Teuchos::RCP< std::vector<panzer::Workset> > worksets_ptr =
    Teuchos::rcp(new std::vector<panzer::Workset>);
  std::vector<panzer::Workset>& worksets = *worksets_ptr;

  // special case for 0 elements!
  if(total_num_cells==0) {

     // Setup integration rules and basis
     RCP<vector<int> > ir_degrees = rcp(new vector<int>(0));
     RCP<vector<string> > basis_names = rcp(new vector<string>(0));

     worksets.resize(1);
     std::vector<panzer::Workset>::iterator i = worksets.begin();
     i->setNumberOfCells(0,0,0);
     i->block_id = elementBlock;
     i->ir_degrees = ir_degrees;
     i->basis_names = basis_names;

     for (std::size_t j=0;j<needs.int_rules.size();j++) {

       RCP<panzer::IntegrationValues2<double> > iv2 =
	 rcp(new panzer::IntegrationValues2<double>("",true));
       iv2->setupArrays(needs.int_rules[j]);

       ir_degrees->push_back(needs.int_rules[j]->cubature_degree);
       i->int_rules.push_back(iv2);
     }

     // Need to create all combinations of basis/ir pairings
     for (std::size_t j=0;j<needs.int_rules.size();j++) {
       for (std::size_t b=0;b<needs.bases.size();b++) {
         RCP<panzer::BasisIRLayout> b_layout
             = rcp(new panzer::BasisIRLayout(needs.bases[b],*needs.int_rules[j]));

         RCP<panzer::BasisValues2<double> > bv2
             = rcp(new panzer::BasisValues2<double>("",true,true));
         bv2->setupArrays(b_layout);
         i->bases.push_back(bv2);

         basis_names->push_back(b_layout->name());
       }

     }

     return worksets_ptr;
  } // end special case

  const int num_verts = node_coordinates.extent(1);
  const int num_coord_dims = node_coordinates.extent(2);
  const int num_dims = cell_topology->getDimension();

  PHX::View<panzer::LocalOrdinal*> workset_local_cell_ids("local_cell_ids",workset_size);
  PHX::View<double***> workset_coords("node_coordinates",workset_size,num_verts,num_coord_dims);

  auto workset_local_cell_ids_h = Kokkos::create_mirror_view(workset_local_cell_ids);

  int offset = 0;
  while(true){

    const int num_cells = std::min(workset_size, total_num_cells-offset);

    for(int i=0; i<num_cells; ++i)
      workset_local_cell_ids_h(i) = local_cell_ids[i+offset];
    Kokkos::deep_copy(workset_local_cell_ids,workset_local_cell_ids_h);

    Kokkos::parallel_for(num_cells, KOKKOS_LAMBDA (int i) {
      for(int j=0; j<num_verts; ++j)
        for(int k=0; k<num_coord_dims; ++k)
          workset_coords(i,j,k) = node_coordinates(i+offset,j,k);
      });
    Kokkos::fence();

    worksets_ptr->push_back(panzer::Workset());
    setupSimpleWorkset(elementBlock, "", side_assembly,
                       subcell_dim, subcell_index, num_cells, cell_topology,
                       workset_local_cell_ids, workset_coords, orientations,
                       worksets_ptr->back());

    offset += num_cells;
    if(offset >= total_num_cells)
      break;
    
  }

  // TODO: This is kept for backward compatability - this path for accessing basis/ir info is deprecated

  // setup the integration rules and bases
  for(std::vector<panzer::Workset>::iterator wkst = worksets.begin(); wkst != worksets.end(); ++wkst)
    populateValueArrays(wkst->num_cells,false,needs,*wkst);

  return worksets_ptr;
}

// ****************************************************************
// ****************************************************************

template<typename ArrayT>
Teuchos::RCP<std::map<unsigned,panzer::Workset> >
panzer::buildBCWorkset(const WorksetNeeds & needs,
                       const std::string & elementBlock,
                       const std::string & sideset,
                       const std::vector<std::size_t>& local_cell_ids,
                       const std::vector<std::size_t>& local_side_ids,
                       const ArrayT& node_coordinates,
                       const bool populate_value_arrays,
                       Teuchos::RCP<const panzer::OrientationsInterface> orientations)
{
  using Teuchos::RCP;
  using Teuchos::rcp;

  panzer::MDFieldArrayFactory mdArrayFactory("",true);

  // key is local face index, value is workset with all elements
  // for that local face
  auto worksets = Teuchos::rcp(new std::map<unsigned,panzer::Workset>);

  // All elements of boundary condition should go into one workset.
  // However due to design of Intrepid2 (requires same basis for all
  // cells), we have to separate the workset based on the local side
  // index.  Each workset for a boundary condition is associated with
  // a local side for the element

  TEUCHOS_ASSERT(local_side_ids.size() == local_cell_ids.size());
  TEUCHOS_ASSERT(local_side_ids.size() == static_cast<std::size_t>(node_coordinates.extent(0)));

  // key is local face index, value is a pair of cell index and vector of element local ids
  std::map< unsigned, std::vector< std::pair< std::size_t, std::size_t>>> element_list;
  for (std::size_t cell = 0; cell < local_cell_ids.size(); ++cell)
    element_list[local_side_ids[cell]].push_back(std::make_pair(cell, local_cell_ids[cell]));

  auto cell_topology = needs.cellData.getCellTopology();
  const int num_verts = node_coordinates.extent(1);
  const int num_coord_dims = node_coordinates.extent(2);
  const int num_dims = cell_topology->getDimension();

  const int oversize_num_cells = local_cell_ids.size();
  PHX::View<panzer::LocalOrdinal*> workset_local_cell_ids("local_cell_ids",oversize_num_cells);
  PHX::View<double***> workset_coords("node_coordinates",oversize_num_cells,num_verts,num_coord_dims);

  auto workset_local_cell_ids_h = Kokkos::create_mirror_view(workset_local_cell_ids);
  auto workset_coords_h = Kokkos::create_mirror_view(workset_coords);

  for (const auto & side_pr : element_list) {

    const int side = side_pr.first;
    const auto & cell_pairs = side_pr.second;

    const size_t num_cells = cell_pairs.size();

    for(int i=0; i<num_cells; ++i){
      workset_local_cell_ids_h(i) = cell_pairs[i].second;
      const int i0 = cell_pairs[i].first;
      for(int j=0; j<num_verts; ++j)
        for(int k=0; k<num_coord_dims; ++k)
          workset_coords_h(i,j,k) = node_coordinates(i0,j,k);
    }
    Kokkos::deep_copy(workset_local_cell_ids,workset_local_cell_ids_h);
    Kokkos::deep_copy(workset_coords,workset_coords_h);
    Kokkos::fence();

    setupSimpleWorkset(elementBlock, sideset, true,
                       num_dims-1, side, num_cells, cell_topology,
                       workset_local_cell_ids, workset_coords, orientations,
                       (*worksets)[side]);
  }

  // setup the integration rules and bases
  if (populate_value_arrays)
    for (auto wkst = worksets->begin(); wkst != worksets->end(); ++wkst) 
      populateValueArrays(wkst->second.num_cells,true,needs,wkst->second);

  return worksets;
}

// ****************************************************************
// ****************************************************************

namespace panzer {
namespace impl {

/* Associate two sets of local side IDs into lists. Each list L has the property
 * that every local side id in that list is the same, and this holds for each
 * local side ID set. The smallest set of lists is found. The motivation for
 * this procedure is to find a 1-1 workset pairing in advance. See the comment
 * re: Intrepid2 in buildBCWorkset for more.
 *   The return value is an RCP to a map. Only the map's values are of interest
 * in practice. Each value is a list L. The map's key is a pair (side ID a, side
 * ID b) that gives rise to the list. We return a pointer to a map so that the
 * caller does not have to deal with the map type; auto suffices.
 */
Teuchos::RCP< std::map<std::pair<std::size_t, std::size_t>, std::vector<std::size_t> > >
associateCellsBySideIds(const std::vector<std::size_t>& sia /* local_side_ids_a */,
                        const std::vector<std::size_t>& sib /* local_side_ids_b */)
{
  TEUCHOS_ASSERT(sia.size() == sib.size());

  auto sip2i_p = Teuchos::rcp(new std::map< std::pair<std::size_t, std::size_t>, std::vector<std::size_t> >);
  auto& sip2i = *sip2i_p;

  for (std::size_t i = 0; i < sia.size(); ++i)
    sip2i[std::make_pair(sia[i], sib[i])].push_back(i);

  return sip2i_p;
}

// Set s = a(idxs). No error checking.
template <typename T>
void subset(const std::vector<T>& a, const std::vector<std::size_t>& idxs, std::vector<T>& s)
{
  s.resize(idxs.size());
  for (std::size_t i = 0; i < idxs.size(); ++i)
    s[i] = a[idxs[i]];
}

template<typename ArrayT>
Teuchos::RCP<std::map<unsigned,panzer::Workset> >
buildBCWorksetForUniqueSideId(const std::string & sideset,
                              const panzer::WorksetNeeds & needs_a,
                              const std::string & blockid_a,
                              const std::vector<std::size_t>& local_cell_ids_a,
                              const std::vector<std::size_t>& local_side_ids_a,
                              const ArrayT& node_coordinates_a,
                              const panzer::WorksetNeeds & needs_b,
                              const std::string & blockid_b,
                              const std::vector<std::size_t>& local_cell_ids_b,
                              const std::vector<std::size_t>& local_side_ids_b,
                              const ArrayT& node_coordinates_b,
                              const WorksetNeeds& needs_b2,
                              Teuchos::RCP<const panzer::OrientationsInterface> orientations)
{
  TEUCHOS_ASSERT(local_cell_ids_a.size() == local_cell_ids_b.size());
  // Get a and b workset maps separately, but don't populate b's arrays.
  const Teuchos::RCP<std::map<unsigned,panzer::Workset> >
    mwa = buildBCWorkset(needs_a,blockid_a, sideset, local_cell_ids_a, local_side_ids_a, node_coordinates_a,true,orientations),
    mwb = buildBCWorkset(needs_b2, blockid_b, sideset, local_cell_ids_b, local_side_ids_b,
                         node_coordinates_b, false /* populate_value_arrays */,orientations);
  TEUCHOS_ASSERT(mwa->size() == 1 && mwb->size() == 1);
  for (std::map<unsigned,panzer::Workset>::iterator ait = mwa->begin(), bit = mwb->begin();
       ait != mwa->end(); ++ait, ++bit) {
    TEUCHOS_ASSERT(Teuchos::as<std::size_t>(ait->second.num_cells) == local_cell_ids_a.size() &&
                   Teuchos::as<std::size_t>(bit->second.num_cells) == local_cell_ids_b.size());
    panzer::Workset& wa = ait->second;
    // Copy b's details(0) to a's details(1).
    wa.other = Teuchos::rcp(new panzer::WorksetDetails(bit->second.details(0)));
    // Populate details(1) arrays so that IP are in order corresponding to details(0).
    populateValueArrays(wa.num_cells, true, needs_b, wa.details(1), Teuchos::rcpFromRef(wa.details(0)));
  }
  // Now mwa has everything we need.
  return mwa;
}

} // namespace impl
} // namespace panzer

// ****************************************************************
// ****************************************************************

template<typename ArrayT>
Teuchos::RCP<std::map<unsigned,panzer::Workset> >
panzer::buildBCWorkset(const std::string & sideset,
                       const WorksetNeeds & needs_a,
                       const std::string & blockid_a,
                       const std::vector<std::size_t>& local_cell_ids_a,
                       const std::vector<std::size_t>& local_side_ids_a,
                       const ArrayT& node_coordinates_a,
                       const panzer::WorksetNeeds & needs_b,
                       const std::string & blockid_b,
                       const std::vector<std::size_t>& local_cell_ids_b,
                       const std::vector<std::size_t>& local_side_ids_b,
                       const ArrayT& node_coordinates_b,
                       Teuchos::RCP<const panzer::OrientationsInterface> orientations)
{
  // Since Intrepid2 requires all side IDs in a workset to be the same (see
  // Intrepid2 comment above), we break the element list into pieces such that
  // each piece contains elements on each side of the interface L_a and L_b and
  // all elemnets L_a have the same side ID, and the same for L_b.
  auto side_id_associations = impl::associateCellsBySideIds(local_side_ids_a, local_side_ids_b);
  if (side_id_associations->size() == 1) {
    // Common case of one workset on each side; optimize for it.
    return impl::buildBCWorksetForUniqueSideId(sideset,
                                               needs_a, blockid_a, local_cell_ids_a, local_side_ids_a, node_coordinates_a,
                                               needs_b, blockid_b, local_cell_ids_b, local_side_ids_b, node_coordinates_b,
                                               needs_b,orientations);
  } else {
    // The interface has elements having a mix of side IDs, so deal with each
    // pair in turn.
    Teuchos::RCP<std::map<unsigned,panzer::Workset> > mwa = Teuchos::rcp(new std::map<unsigned,panzer::Workset>);
    std::vector<std::size_t> lci_a, lci_b, lsi_a, lsi_b;
    panzer::MDFieldArrayFactory mdArrayFactory("", true);
    const int d1 = Teuchos::as<std::size_t>(node_coordinates_a.extent(1)),
      d2 = Teuchos::as<std::size_t>(node_coordinates_a.extent(2));
    for (auto it : *side_id_associations) {
      const auto& idxs = it.second;
      impl::subset(local_cell_ids_a, idxs, lci_a);
      impl::subset(local_side_ids_a, idxs, lsi_a);
      impl::subset(local_cell_ids_b, idxs, lci_b);
      impl::subset(local_side_ids_b, idxs, lsi_b);
      auto nc_a = mdArrayFactory.buildStaticArray<double,Cell,NODE,Dim>("nc_a", idxs.size(), d1, d2);
      auto nc_b = mdArrayFactory.buildStaticArray<double,Cell,NODE,Dim>("nc_b", idxs.size(), d1, d2);
      auto nc_a_h = Kokkos::create_mirror_view(nc_a.get_static_view());
      auto nc_b_h = Kokkos::create_mirror_view(nc_b.get_static_view());
      auto node_coordinates_a_h = Kokkos::create_mirror_view(PHX::as_view(node_coordinates_a));
      auto node_coordinates_b_h = Kokkos::create_mirror_view(PHX::as_view(node_coordinates_b));
      Kokkos::deep_copy(node_coordinates_a_h, PHX::as_view(node_coordinates_a));
      Kokkos::deep_copy(node_coordinates_b_h, PHX::as_view(node_coordinates_b));
      for (std::size_t i = 0; i < idxs.size(); ++i) {
        const auto ii = idxs[i];
        for (int j = 0; j < d1; ++j)
          for (int k = 0; k < d2; ++k) {
            nc_a_h(i, j, k) = node_coordinates_a_h(ii, j, k);
            nc_b_h(i, j, k) = node_coordinates_b_h(ii, j, k);
          }
      }
      Kokkos::deep_copy(nc_a.get_static_view(), nc_a_h);
      Kokkos::deep_copy(nc_b.get_static_view(), nc_b_h);
      auto mwa_it = impl::buildBCWorksetForUniqueSideId(sideset,
                                                        needs_a,blockid_a, lci_a, lsi_a, nc_a,
                                                        needs_b,blockid_b, lci_b, lsi_b, nc_b,
                                                        needs_b,orientations);
      TEUCHOS_ASSERT(mwa_it->size() == 1);
      // Form a unique key that encodes the pair (side ID a, side ID b). We
      // abuse the key here in the sense that it is everywhere else understood
      // to correspond to the side ID of the elements in the workset.
      //   1000 is a number substantially larger than is needed for any element.
      const std::size_t key = lsi_a[0] * 1000 + lsi_b[0];
      (*mwa)[key] = mwa_it->begin()->second;
    }
    return mwa;
  }
}

// ****************************************************************
// ****************************************************************

template<typename ArrayT>
Teuchos::RCP<std::vector<panzer::Workset> >
panzer::buildEdgeWorksets(const WorksetNeeds & needs_a,
                   const std::string & eblock_a,
	 	               const std::vector<std::size_t>& local_cell_ids_a,
		               const std::vector<std::size_t>& local_side_ids_a,
		               const ArrayT& node_coordinates_a,
                   const WorksetNeeds & needs_b,
                   const std::string & eblock_b,
		               const std::vector<std::size_t>& local_cell_ids_b,
		               const std::vector<std::size_t>& local_side_ids_b,
		               const ArrayT& node_coordinates_b)
{
  using Teuchos::RCP;
  using Teuchos::rcp;

  panzer::MDFieldArrayFactory mdArrayFactory("",true);

  std::size_t total_num_cells_a = local_cell_ids_a.size();
  std::size_t total_num_cells_b = local_cell_ids_b.size();

  TEUCHOS_ASSERT(total_num_cells_a==total_num_cells_b);
  TEUCHOS_ASSERT(local_side_ids_a.size() == local_cell_ids_a.size());
  TEUCHOS_ASSERT(local_side_ids_a.size() == static_cast<std::size_t>(node_coordinates_a.extent(0)));
  TEUCHOS_ASSERT(local_side_ids_b.size() == local_cell_ids_b.size());
  TEUCHOS_ASSERT(local_side_ids_b.size() == static_cast<std::size_t>(node_coordinates_b.extent(0)));

  std::size_t total_num_cells = total_num_cells_a;

  std::size_t workset_size = needs_a.cellData.numCells();

  Teuchos::RCP< std::vector<panzer::Workset> > worksets_ptr =
    Teuchos::rcp(new std::vector<panzer::Workset>);
  std::vector<panzer::Workset>& worksets = *worksets_ptr;

  // special case for 0 elements!
  if(total_num_cells==0) {

     // Setup integration rules and basis
     RCP<std::vector<int> > ir_degrees = rcp(new std::vector<int>(0));
     RCP<std::vector<std::string> > basis_names = rcp(new std::vector<std::string>(0));

     worksets.resize(1);
     std::vector<panzer::Workset>::iterator i = worksets.begin();

     i->details(0).block_id = eblock_a;
     i->other = Teuchos::rcp(new panzer::WorksetDetails);
     i->details(1).block_id = eblock_b;

     i->num_cells = 0;
     i->ir_degrees = ir_degrees;
     i->basis_names = basis_names;

     for(std::size_t j=0;j<needs_a.int_rules.size();j++) {

      RCP<panzer::IntegrationValues2<double> > iv2 =
         rcp(new panzer::IntegrationValues2<double>("",true));
       iv2->setupArrays(needs_a.int_rules[j]);

       ir_degrees->push_back(needs_a.int_rules[j]->cubature_degree);
       i->int_rules.push_back(iv2);
     }

     // Need to create all combinations of basis/ir pairings
     for(std::size_t j=0;j<needs_a.int_rules.size();j++) {

        for(std::size_t b=0;b<needs_a.bases.size();b++) {

	 RCP<panzer::BasisIRLayout> b_layout = rcp(new panzer::BasisIRLayout(needs_a.bases[b],*needs_a.int_rules[j]));

	 RCP<panzer::BasisValues2<double> > bv2 =
	   rcp(new panzer::BasisValues2<double>("",true,true));
	 bv2->setupArrays(b_layout);
	 i->bases.push_back(bv2);

	 basis_names->push_back(b_layout->name());
       }

     }

     return worksets_ptr;
  } // end special case

  // This collects all the elements that share the same sub cell pairs, this makes it easier to
  // build the required worksets
  // key is the pair of local face indices, value is a vector of cell indices that satisfy this pair
  std::map<std::pair<unsigned,unsigned>,std::vector<std::size_t> > element_list;
  for (std::size_t cell=0; cell < local_cell_ids_a.size(); ++cell)
    element_list[std::make_pair(local_side_ids_a[cell],local_side_ids_b[cell])].push_back(cell);

  // figure out how many worksets will be needed, resize workset vector accordingly
  std::size_t num_worksets = 0;
  for(const auto& edge : element_list) {
    std::size_t num_worksets_for_edge = edge.second.size() / workset_size;
    std::size_t last_workset_size = edge.second.size() % workset_size;
    if(last_workset_size!=0)
      num_worksets_for_edge += 1;

    num_worksets += num_worksets_for_edge;
  }
  worksets.resize(num_worksets);

  // fill the worksets
  std::vector<Workset>::iterator current_workset = worksets.begin();
  for(const auto& edge : element_list) {
    // loop over each workset
    const std::vector<std::size_t> & cell_indices = edge.second;

    current_workset = buildEdgeWorksets(cell_indices,
                                       needs_a,eblock_a,local_cell_ids_a,local_side_ids_a,node_coordinates_a,
                                       needs_b,eblock_b,local_cell_ids_b,local_side_ids_b,node_coordinates_b,
                                       current_workset);
  }

  // sanity check
  TEUCHOS_ASSERT(current_workset==worksets.end());

  return worksets_ptr;
}

template<typename ArrayT>
std::vector<panzer::Workset>::iterator
panzer::buildEdgeWorksets(const std::vector<std::size_t> & cell_indices,
                          const WorksetNeeds & needs_a,
                          const std::string & eblock_a,
	 	                      const std::vector<std::size_t>& local_cell_ids_a,
		                      const std::vector<std::size_t>& local_side_ids_a,
		                      const ArrayT& node_coordinates_a,
                          const WorksetNeeds & needs_b,
                          const std::string & eblock_b,
	      	                const std::vector<std::size_t>& local_cell_ids_b,
		                      const std::vector<std::size_t>& local_side_ids_b,
		                      const ArrayT& node_coordinates_b,
                          std::vector<Workset>::iterator beg)
{
  panzer::MDFieldArrayFactory mdArrayFactory("",true);

  std::vector<Workset>::iterator wkst = beg;

  std::size_t current_cell_index = 0;
  while (current_cell_index<cell_indices.size()) {
    std::size_t workset_size = needs_a.cellData.numCells();

    // allocate workset details (details(0) is already associated with the
    // workset object itself)
    wkst->other = Teuchos::rcp(new panzer::WorksetDetails);

    wkst->subcell_dim = needs_a.cellData.baseCellDimension()-1;

    wkst->details(0).subcell_index = local_side_ids_a[cell_indices[current_cell_index]];
    wkst->details(0).block_id = eblock_a;
    wkst->details(0).cell_node_coordinates = mdArrayFactory.buildStaticArray<double,Cell,NODE,Dim>("cnc",workset_size,
					 node_coordinates_a.extent(1),
					 node_coordinates_a.extent(2));

    wkst->details(1).subcell_index = local_side_ids_b[cell_indices[current_cell_index]];
    wkst->details(1).block_id = eblock_b;
    wkst->details(1).cell_node_coordinates = mdArrayFactory.buildStaticArray<double,Cell,NODE,Dim>("cnc",workset_size,
					 node_coordinates_a.extent(1),
					 node_coordinates_a.extent(2));

    std::size_t remaining_cells = cell_indices.size()-current_cell_index;
    if(remaining_cells<workset_size)
      workset_size = remaining_cells;

    // this is the true number of cells in this workset
    wkst->setNumberOfCells(workset_size,0,0);
    wkst->details(0).cell_local_ids.resize(workset_size);
    wkst->details(1).cell_local_ids.resize(workset_size);

    auto dim0_cell_node_coordinates_view = wkst->details(0).cell_node_coordinates.get_static_view();
    auto dim0_cell_node_coordinates_h = Kokkos::create_mirror_view(dim0_cell_node_coordinates_view);
    Kokkos::deep_copy(dim0_cell_node_coordinates_h, dim0_cell_node_coordinates_view);

    auto dim1_cell_node_coordinates_view = wkst->details(1).cell_node_coordinates.get_static_view();
    auto dim1_cell_node_coordinates_h = Kokkos::create_mirror_view(dim1_cell_node_coordinates_view);
    Kokkos::deep_copy(dim1_cell_node_coordinates_h, dim1_cell_node_coordinates_view);

    auto node_coordinates_a_h = Kokkos::create_mirror_view(node_coordinates_a);
    Kokkos::deep_copy(node_coordinates_a_h, node_coordinates_a);

    auto node_coordinates_b_h = Kokkos::create_mirror_view(node_coordinates_b);
    Kokkos::deep_copy(node_coordinates_b_h, node_coordinates_b);

    for(std::size_t cell=0;cell<workset_size; cell++,current_cell_index++) {

      wkst->details(0).cell_local_ids[cell] = local_cell_ids_a[cell_indices[current_cell_index]];
      wkst->details(1).cell_local_ids[cell] = local_cell_ids_b[cell_indices[current_cell_index]];

      for (std::size_t node = 0; node < Teuchos::as<std::size_t>(node_coordinates_a.extent(1)); ++ node) {
	      for (std::size_t dim = 0; dim < Teuchos::as<std::size_t>(node_coordinates_a.extent(2)); ++ dim) {
          dim0_cell_node_coordinates_h(cell,node,dim) = node_coordinates_a_h(cell_indices[current_cell_index],node,dim);
          dim1_cell_node_coordinates_h(cell,node,dim) = node_coordinates_b_h(cell_indices[current_cell_index],node,dim);
        }
      }
    }

    Kokkos::deep_copy(dim0_cell_node_coordinates_view, dim0_cell_node_coordinates_h);
    Kokkos::deep_copy(dim1_cell_node_coordinates_view, dim1_cell_node_coordinates_h);

    auto cell_local_ids_k_0 = PHX::View<int*>("Workset:cell_local_ids",wkst->details(0).cell_local_ids.size());
    auto cell_local_ids_k_0_h = Kokkos::create_mirror_view(cell_local_ids_k_0);

    auto cell_local_ids_k_1 = PHX::View<int*>("Workset:cell_local_ids",wkst->details(1).cell_local_ids.size());
    auto cell_local_ids_k_1_h = Kokkos::create_mirror_view(cell_local_ids_k_1);

    for(std::size_t i=0;i<wkst->details(0).cell_local_ids.size();i++)
      cell_local_ids_k_0_h(i) = wkst->details(0).cell_local_ids[i];
    for(std::size_t i=0;i<wkst->details(1).cell_local_ids.size();i++)
      cell_local_ids_k_1_h(i) = wkst->details(1).cell_local_ids[i];

    Kokkos::deep_copy(cell_local_ids_k_0, cell_local_ids_k_0_h);
    Kokkos::deep_copy(cell_local_ids_k_1, cell_local_ids_k_1_h);

    wkst->details(0).cell_local_ids_k = cell_local_ids_k_0;
    wkst->details(1).cell_local_ids_k = cell_local_ids_k_1;

    // fill the BasisValues and IntegrationValues arrays
    std::size_t max_workset_size = needs_a.cellData.numCells();
    populateValueArrays(max_workset_size,true,needs_a,wkst->details(0)); // populate "side" values
    populateValueArrays(max_workset_size,true,needs_b,wkst->details(1),Teuchos::rcpFromRef(wkst->details(0)));

    wkst++;
  }

  return wkst;
}

#endif
