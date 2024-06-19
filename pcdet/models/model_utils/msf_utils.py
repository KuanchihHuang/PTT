import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from typing import Optional, List
from torch import Tensor
from torch.nn.init import xavier_uniform_, zeros_, kaiming_normal_
from spconv.pytorch.utils import PointToVoxel

class VoxelSampler(nn.Module):
    GAMMA = 1.1
    def __init__(self, device, voxel_size, pc_range, max_points_per_voxel, num_point_features=5):
        super().__init__()
        
        self.voxel_size = voxel_size
        
        self.gen = PointToVoxel(
                        vsize_xyz=[voxel_size, voxel_size, pc_range[5]-pc_range[2]],
                        coors_range_xyz=pc_range,
                        num_point_features=num_point_features, 
                        max_num_voxels=50000,
                        max_num_points_per_voxel=max_points_per_voxel,
                        device=device
                    )
        
        self.pc_start = torch.FloatTensor( pc_range[:2] ).to(device)
        self.k = max_points_per_voxel
        self.grid_x = int((pc_range[3] - pc_range[0]) / voxel_size)
        self.grid_y = int((pc_range[4] - pc_range[1]) / voxel_size)
    def get_output_feature_dim(self):
        return self.num_point_features

    @staticmethod
    def cylindrical_pool(cur_points, cur_boxes, num_sample, gamma=1.):   
        if len(cur_points) < num_sample:
            cur_points = F.pad(cur_points, [0, 0, 0, num_sample-len(cur_points)])
        cur_radiis = torch.norm(cur_boxes[:, 3:5]/2, dim=-1) * gamma
        dis = torch.norm((cur_points[:,:2].unsqueeze(0) - cur_boxes[:,:2].unsqueeze(1).repeat(1,cur_points.shape[0],1)), dim = 2)
        point_mask = (dis <= cur_radiis.unsqueeze(-1))

        sampled_mask, sampled_idx = torch.topk(point_mask.float(), num_sample)
        sampled_idx = sampled_idx.view(-1, 1).repeat(1, cur_points.shape[-1])
        sampled_points = torch.gather(cur_points, 0, sampled_idx).view(len(sampled_mask), num_sample, -1)

        sampled_points[sampled_mask==0, :] = 0

        return sampled_points


    def forward(self, batch_size, trajectory_rois, num_sample, batch_dict): 
        
        src = list()
        for bs_idx in range(batch_size):
            
            cur_points = batch_dict['points'][(batch_dict['points'][:, 0] == bs_idx)][:,1:]
            cur_batch_boxes = trajectory_rois[bs_idx]
            src_points = list()
            for idx in range(trajectory_rois.shape[1]):
                gamma = self.GAMMA # ** (idx+1)

                time_mask = (cur_points[:,-1] - idx*0.1).abs() < 1e-3
                cur_time_points = cur_points[time_mask, :5].contiguous()

                cur_frame_boxes = cur_batch_boxes[idx]

                voxel, coords, num_points = self.gen(cur_time_points) 

                coords = coords[:, [2, 1]].contiguous()

                query_coords = ( cur_frame_boxes[:, :2] - self.pc_start ) // self.voxel_size
    
                radiis = torch.ceil( 
                   torch.norm(cur_frame_boxes[:, 3:5]/2, dim=-1) * gamma / self.voxel_size )


                # h_table = torch.zeros(self.grid_x*self.grid_y).fill_(-1).to(coords)
                # coords_ = coords[:, 0] * self.grid_y + coords[:, 1]
                # h_table[coords_.long()] = torch.arange(len(coords)).to(coords)
                
                # v_indice = torch.zeros((len(query_coords)), int(radiis.max())**2).fill_(-1).to(coords)
                # scatter.hash_query(self.grid_x, self.grid_y, query_coords.int(), radiis.int(), h_table, v_indice)
                # v_indice = v_indice.long()

                # voxel_points = voxel[v_indice, :, :]
                # num_points = num_points[v_indice, None]

                # cur_radiis = torch.norm(cur_frame_boxes[:, None, None, 3:5]/2, dim=-1) * gamma
                # dis = torch.norm(voxel_points[:, :, :, :2] - cur_frame_boxes[:, None, None, :2], dim = -1)
                # point_mask = dis <= cur_radiis

                # a, b, _ = num_points.shape
                # points_mask = point_mask & (v_indice[:, :, None]!=-1) & \
                #     (num_points > torch.arange(self.k)[None, None, :].repeat(a, b, 1).type_as(num_points))
                
                # points_mask = points_mask.flatten(1, 2)
                # voxel_points = voxel_points.flatten(1, 2)

                # random_perm = torch.randperm(points_mask.shape[1])
                # points_mask = points_mask[:, random_perm]
                # voxel_points = voxel_points[:, random_perm, :]

                
                # try:
                #     sampled_mask, sampled_idx = torch.topk(points_mask.float(), num_sample)
           
                #     key_points = torch.gather(voxel_points, 1, sampled_idx[:, :, None].repeat(1, 1, voxel_points.shape[-1]))
                #     key_points[sampled_mask==0, :] = 0
                # except:
                #     key_points = torch.zeros([len(cur_frame_boxes), num_sample, 5]).to(voxel)
                #     key_points[:, :voxel_points.shape[1], :] = voxel_points

                dist = torch.abs(query_coords[:, None, :2] - coords[None, :, :] )

                voxel_mask = torch.all(dist < radiis[:, None, None], dim=-1).any(0)
            
                num_points = num_points[voxel_mask]
                key_points = voxel[voxel_mask, :]

                point_mask = torch.arange(self.k)[None, :].repeat(len(key_points), 1).type_as(num_points)

                point_mask = num_points[: , None] > point_mask
                key_points = key_points[point_mask]
                key_points = key_points[ torch.randperm(len(key_points)), :]
                
                key_points = self.cylindrical_pool(key_points, cur_frame_boxes, num_sample, gamma)

                src_points.append( key_points )
                
            src.append(torch.stack(src_points))

        return torch.stack(src).permute(0, 2, 1, 3, 4).flatten(2, 3)
        

def build_voxel_sampler(device):
    return VoxelSampler(
        device,
        voxel_size=0.4,
        pc_range=[-75.2, -75.2, -10, 75.2, 75.2, 10],
        max_points_per_voxel=32,
        num_point_features=5
    )


