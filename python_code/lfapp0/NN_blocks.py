## Define the convolutional neural network architecture
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I

dim0=32



class PRT_EMB16(nn.Module):
    def __init__(self, state_dim=2, embed_dim_per_group=128, fc_groups=1, add_nof_targs=False, inner_layers_width_mult=1, nof_layers_split_to_groups=100, nof_fc_layers=5):
        super(PRT_EMB16, self).__init__()
        self.state_dim = state_dim
        self.add_nof_targs = add_nof_targs
        qq_in = self.state_dim + 1 + self.add_nof_targs*2
        self.embed_dim_per_group = embed_dim_per_group
        self.fc_groups = fc_groups#nof_targs*(1+bool(self.add_baseline_locs))
        self.fc_groups_fcl_num = np.ones(nof_fc_layers+1, dtype=int)*fc_groups#nof_targs*(1+bool(self.add_baseline_locs))
        self.fc_groups_fcl_num[0] = 0
        self.nof_layers_split_to_groups = np.minimum(nof_fc_layers, nof_layers_split_to_groups)
        self.nof_fc_layers = nof_fc_layers
        for layer_num in np.arange(1,self.nof_fc_layers+1, 1):
            if layer_num <= self.nof_fc_layers-self.nof_layers_split_to_groups:
                self.fc_groups_fcl_num[layer_num] = 1
        self.inner_layers_width_mult = inner_layers_width_mult
        out_features = self.embed_dim_per_group*self.fc_groups

        fc1_in_channels  = qq_in
        fc2_in_channels  = self.inner_layers_width_mult*int(self.embed_dim_per_group/1)*self.fc_groups_fcl_num[1]
        fc3_in_channels  = self.inner_layers_width_mult*int(self.embed_dim_per_group/1)*self.fc_groups_fcl_num[2]
        fc4_in_channels  = self.inner_layers_width_mult*int(self.embed_dim_per_group/1)*self.fc_groups_fcl_num[3]
        temp=0
        if self.nof_fc_layers == 6:
            temp = 1
            fc4_1_in_channels = self.inner_layers_width_mult * int(self.embed_dim_per_group / 1) * self.fc_groups_fcl_num[4]
            self.fc4_1_in_channels = fc4_1_in_channels
        fc5_in_channels  = self.inner_layers_width_mult*int(self.embed_dim_per_group/1)*self.fc_groups_fcl_num[4+temp]


        fc1_out_channels = fc2_in_channels
        fc2_out_channels = fc3_in_channels
        fc3_out_channels = fc4_in_channels
        if not self.nof_fc_layers == 6:
            fc4_out_channels = fc5_in_channels
        else:
            fc4_out_channels = fc4_1_in_channels
            fc4_1_out_channels = fc5_in_channels
        fc5_out_channels = out_features

        self.fc1_in_channels = fc1_in_channels
        self.fc2_in_channels = fc2_in_channels
        self.fc3_in_channels = fc3_in_channels
        self.fc4_in_channels = fc4_in_channels
        self.fc5_in_channels = fc5_in_channels

        #self.fc1 = nn.Linear(fc1_in_channels, fc1_out_channels, bias=True)
        self.fc1 = nn.Conv2d(in_channels=fc1_in_channels, out_channels=fc1_out_channels, kernel_size=1, groups=1, bias=True)
        self.fc2 = nn.Conv2d(in_channels=fc2_in_channels, out_channels=fc2_out_channels, kernel_size=1, groups=self.fc_groups_fcl_num[2], bias=True)
        self.fc3 = nn.Conv2d(in_channels=fc3_in_channels, out_channels=fc3_out_channels, kernel_size=1, groups=self.fc_groups_fcl_num[3], bias=True)
        self.fc4 = nn.Conv2d(in_channels=fc4_in_channels, out_channels=fc4_out_channels, kernel_size=1, groups=self.fc_groups_fcl_num[4], bias=True)
        if self.nof_fc_layers == 6:
            self.fc4_1 = nn.Conv2d(in_channels=fc4_1_in_channels, out_channels=fc4_1_out_channels, kernel_size=1, groups=self.fc_groups_fcl_num[5], bias=True)
        self.fc5 = nn.Conv2d(in_channels=fc5_in_channels, out_channels=fc5_out_channels, kernel_size=1, groups=self.fc_groups_fcl_num[5+temp], bias=True)

        #self.activation = torch.nn.ELU()
        self.activation = torch.nn.LeakyReLU()

    def forward(self, x0):

        try:
            assert torch.all(torch.isnan(x0) == False)
        except:
            sfdasfda = 5
            assert torch.all(torch.isnan(x0) == False)
        # ddd = self.model_att.nn3.particle_emb.fc1.weight.detach().cpu().numpy()
        x = self.activation(self.fc1(x0))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        if self.nof_fc_layers == 6:
            x = self.activation(self.fc4_1(x))
        x = self.fc5(x)
        return x


class SDP16(nn.Module):
    def __init__(self, particle_embed_dim=dim0, q_head_out_dim=dim0, v_head_out_dim=dim0, another_layer = True, nof_net_types=2, qkv_nof_additional_fc=0, fc_inner_layers_width_mult=1):
        # same as SA1_007 but with seperated embeddings for each trgat
        super(SDP16, self).__init__()
        #self.batch_size = batch_size
        self.nof_net_types = nof_net_types
        self.v1_out_dim = v_head_out_dim
        self.particle_embed_dim = particle_embed_dim
        self.output_wts = 1
        self.output_locs = 1
        self.wts_and_or_locs = self.output_wts+self.output_locs
        #self.nof_types = 3
        self.fc_inner_layers_width_mult = fc_inner_layers_width_mult
        self.q_head_in_dim = self.particle_embed_dim
        self.k_head_in_dim = self.q_head_in_dim
        self.v_head_in_dim = self.particle_embed_dim
        self.fc_groups_per_network = self.nof_net_types
        self.v_head_out_dim = v_head_out_dim
        self.qkv_nof_additional_fc = qkv_nof_additional_fc
        self.q_head_out_dim = q_head_out_dim
        self.k_head_out_dim = self.q_head_out_dim
        self.first_linear_out_dim=v_head_out_dim

        self.do_concat = True
        if self.do_concat:
            self.W_h1_in_per_group = (self.particle_embed_dim + self.v_head_out_dim)
        else:
            self.W_h1_in_per_group = self.particle_embed_dim
        self.qkv_nof_types = 3


        self.net_qkv = NET1_QKV16(head_in_dim=self.q_head_in_dim, head_out_dim=self.q_head_out_dim, qkv_nof_types=self.qkv_nof_types, nof_net_types=self.nof_net_types, nof_additional_fc=self.qkv_nof_additional_fc)

        # joind heads keeps targets seperated groups=self.nof_targs
        W_h_out_channels =  self.fc_groups_per_network * self.v_head_out_dim * self.fc_inner_layers_width_mult
        self.W_h1 = nn.Conv2d(in_channels=self.W_h1_in_per_group * self.fc_groups_per_network, out_channels=W_h_out_channels, kernel_size=1, groups=self.fc_groups_per_network, bias=True)
        #W_h1_0_out_channels = self.fc_groups_per_network * self.v_head_out_dim if not self.another_layer else self.fc_groups_per_network * self.v_head_out_dim*self.fc_inner_layers_width_mult
        self.W_h1_0 = nn.Conv2d(in_channels=W_h_out_channels, out_channels=W_h_out_channels, kernel_size=1, groups=self.fc_groups_per_network, bias=True)
        self.another_layer = another_layer
        self.another_another_layer = 0
        if self.another_layer:#88
            self.W_h1_1 = nn.Conv2d(in_channels=W_h_out_channels, out_channels=W_h_out_channels, kernel_size=1, groups=self.fc_groups_per_network, bias=True)
            if self.another_another_layer:#89
                self.W_h1_2 = nn.Conv2d(in_channels=W_h_out_channels, out_channels=W_h_out_channels, kernel_size=1, groups=self.fc_groups_per_network, bias=True)
        assert self.particle_embed_dim % self.q_head_in_dim == 0, "d_model %  should be zero."
        self.activation_wts = torch.nn.LeakyReLU()

    def forward(self, x_e):
        batch_size, nof_parts, in_dim = x_e.shape
        # in_dim = nof_targs*nof_nets*dim
        # batch|particle        |emb_dim*nof_networks*nof_targs|1|1
        # b1,b2..|,p1,p2..|emb1,emb2,..                              |1|1
        x_e = torch.reshape(x_e, (batch_size * nof_parts, in_dim, 1, 1))
        # batch*particle        |emb_dim*nof_networks*fc_groups_per_network|1|1
        # b1p1,b1p2..,b2p1,b2p2..|emb1,emb2,..                              |1|1
        qkv = self.net_qkv(x_e)
        qkv2 = torch.reshape(qkv, (batch_size, nof_parts, self.nof_net_types, self.qkv_nof_types, 1,self.particle_embed_dim))
        qkv3 = torch.permute(qkv2, (0, 2, 3, 4, 1, 5))
        qs = qkv3[:, :, 0]
        ks = qkv3[:, :, 1]
        vs = qkv3[:, :, 2]
        q = torch.reshape(qs, (batch_size * self.nof_net_types, nof_parts, self.q_head_out_dim))
        k = torch.reshape(ks, (batch_size * self.nof_net_types, nof_parts, self.q_head_out_dim))
        k = torch.permute(k, (0, 2, 1)) / torch.sqrt(torch.tensor(self.k_head_out_dim))  # (bs, n_heads, q_length, dim_per_head)
        ip = torch.bmm(q, k)
        sip = torch.softmax(ip / torch.sqrt(torch.tensor(self.k_head_out_dim)), dim=-1)
        v = torch.reshape(vs, (batch_size * self.nof_net_types, nof_parts, self.q_head_out_dim))
        x = torch.bmm(sip, v)
        x = x.reshape((batch_size, self.nof_net_types, nof_parts, self.v_head_out_dim))
        v_before = torch.reshape(vs, (batch_size, self.nof_net_types, nof_parts, self.v_head_out_dim))
        if self.do_concat:
            #x_xe = torch.cat((x_e, x), dim=3)
            x_xe = torch.cat((v_before, x), dim=3)
        else:
            x_xe = x-v_before
        x_xe = torch.permute(x_xe, (0, 2, 1, 3))  # the order 2,1 and not 1,2 to get the different heads of the same target to be close for the conv2 groups
        x_xe = torch.reshape(x_xe, (batch_size* nof_parts , self.nof_net_types*self.W_h1_in_per_group, 1, 1))

        x = self.W_h1(x_xe)  # (bs, q_length, dim)
        x = self.activation_wts(x)
        x = self.activation_wts(self.W_h1_0(x))
        if self.another_layer:
            x = self.activation_wts(self.W_h1_1(x))
            if self.another_another_layer:
                x = self.activation_wts(self.W_h1_2(x))
        return x


class NET1_QKV16(nn.Module):
    def __init__(self, head_in_dim, head_out_dim, qkv_nof_types, nof_net_types=2, nof_additional_fc=0):
        super(NET1_QKV16, self).__init__()
        self.head_in_dim = head_in_dim
        self.head_out_dim = head_out_dim
        # for each target get 3(q, k, v)*1(heads) outputs
        self.qkv_nof_types = qkv_nof_types
        self.nof_net_types = nof_net_types
        self.fc_groups = self.qkv_nof_types*self.nof_net_types
        fc1_in_channels  = self.nof_net_types*self.head_in_dim
        fc1_out_channels = self.fc_groups*self.head_out_dim
        self.nof_additional_fc = nof_additional_fc
        if self.nof_additional_fc==1:
            fc2_out_channels = fc1_out_channels
            fc1_out_channels = fc1_out_channels*4
            fc2_in_channels = fc1_out_channels
            self.fc2 = nn.Conv2d(in_channels=fc2_in_channels, out_channels=fc2_out_channels, kernel_size=1, groups=self.nof_net_types, bias=True)
        self.fc1 = nn.Conv2d(in_channels=fc1_in_channels, out_channels=fc1_out_channels, kernel_size=1, groups=self.nof_net_types, bias=True)
        self.activation = torch.nn.LeakyReLU()

    def forward(self, x0):
        batch_size = x0.shape[0]

        x = self.fc1(x0)
        if self.nof_additional_fc==1:
            x = self.activation(x)
            x = self.fc2(x)

        return x



class SA1_016(nn.Module):
    def __init__(self, skip=False, state_dim=2, particle_embed_dim=2*dim0, q_head_out_dim=2*dim0, v_head_out_dim=2*dim0, add_baseline_locs=1, another_sdp=True, do_sat_part_emb=True, add_nof_targs=False, emb_per_sub_net=True, emb_per_sup_net=True, emb_inner_layers_width_mult=1, fc_inner_layers_width_mult=1, emb_nof_layers_split_to_groups=100, emb_nof_fc_layers=5, qkv_nof_additional_fc=0, baseline_is_wtd_avg=False, additional_flock_for_avg=False, divide_by_wts=True):
        # same as SA1_007 but with seperated embeddings for each trgat4
        super(SA1_016, self).__init__()
        self.another_sdp = another_sdp
        self.skip = skip
        self.v1_out_dim = v_head_out_dim
        self.particle_embed_dim = particle_embed_dim
        self.divide_by_wts = divide_by_wts
        self.output_wts = 1
        self.output_locs = 1
        self.wts_and_or_locs = self.output_wts+self.output_locs
        self.add_nof_targs = add_nof_targs
        self.state_dim = state_dim
        self.embed_in_dim = self.state_dim + 1 + self.add_nof_targs*2
        self.fc_inner_layers_width_mult = fc_inner_layers_width_mult
        self.baseline_is_wtd_avg = baseline_is_wtd_avg
        self.additional_flock_for_avg = additional_flock_for_avg
        #self.nof_types = 3
        self.q_head_in_dim = self.particle_embed_dim
        self.k_head_in_dim = self.q_head_in_dim
        self.v_head_in_dim = self.particle_embed_dim
        self.add_baseline_locs = add_baseline_locs
        self.nof_sub_networks = 1+bool(self.add_baseline_locs)+bool(self.additional_flock_for_avg)
        self.do_sat_part_emb = do_sat_part_emb
        self.qkv_nof_additional_fc = qkv_nof_additional_fc
        self.nof_sup_networks = 2 if self.do_sat_part_emb else 1
        self.v_head_out_dim = v_head_out_dim
        self.q_head_out_dim = q_head_out_dim
        self.k_head_out_dim = self.q_head_out_dim
        self.out_dim_wts = 1
        self.first_linear_out_dim=v_head_out_dim
        self.W_h4_output_dim = 1
        self.post_sdp_groups = self.nof_sub_networks
        self.out_dim_locs = self.state_dim #if
        self.emb_per_sub_net = emb_per_sub_net
        self.emb_per_sup_net = emb_per_sup_net
        self.sub_n = self.nof_sub_networks if self.emb_per_sub_net else 1
        self.sup_n = self.nof_sup_networks if self.emb_per_sup_net else 1
        self.emb_fc_groups = self.sub_n*self.sup_n
        self.part_emb_dim = self.particle_embed_dim
        self.particle_emb16 = PRT_EMB16(self.state_dim, embed_dim_per_group=self.part_emb_dim, fc_groups=self.emb_fc_groups, add_nof_targs=self.add_nof_targs, inner_layers_width_mult=emb_inner_layers_width_mult, nof_layers_split_to_groups=emb_nof_layers_split_to_groups, nof_fc_layers=emb_nof_fc_layers)
        self.sdp16_1 = SDP16(particle_embed_dim=particle_embed_dim, q_head_out_dim=q_head_out_dim, v_head_out_dim=v_head_out_dim, another_layer=True, nof_net_types=self.nof_sub_networks, qkv_nof_additional_fc=self.qkv_nof_additional_fc, fc_inner_layers_width_mult=self.fc_inner_layers_width_mult)
        if self.another_sdp:
            self.W_h2 = nn.Conv2d(in_channels=self.fc_inner_layers_width_mult*self.nof_sub_networks*self.v_head_out_dim, out_channels= self.nof_sub_networks*self.first_linear_out_dim, kernel_size=1, groups=self.nof_sub_networks, bias=True)
            self.sdp16_2 = SDP16(particle_embed_dim=particle_embed_dim, q_head_out_dim=q_head_out_dim, v_head_out_dim=v_head_out_dim, another_layer=True, nof_net_types=self.nof_sub_networks, qkv_nof_additional_fc=self.qkv_nof_additional_fc, fc_inner_layers_width_mult=self.fc_inner_layers_width_mult)
        # for each particle
        self.W_h3_W_h4_groups = 1
        if self.add_baseline_locs and self.baseline_is_wtd_avg:
            self.W_h3_W_h4_groups +=1
        if self.additional_flock_for_avg:
            self.W_h3_W_h4_groups += 1

        self.W_h3 = nn.Conv2d(in_channels=self.W_h3_W_h4_groups*self.v_head_out_dim*self.fc_inner_layers_width_mult, out_channels=self.W_h3_W_h4_groups*self.first_linear_out_dim*self.fc_inner_layers_width_mult, kernel_size=1, groups=self.W_h3_W_h4_groups, bias=True)
        self.W_h4 = nn.Conv2d(in_channels=self.W_h3_W_h4_groups*self.first_linear_out_dim*self.fc_inner_layers_width_mult, out_channels=self.W_h3_W_h4_groups*(self.out_dim_locs+self.out_dim_wts), kernel_size=1, groups=self.W_h3_W_h4_groups, bias=True)
        # for average particle
        if (not self.baseline_is_wtd_avg and not self.additional_flock_for_avg) or (self.add_baseline_locs and not self.baseline_is_wtd_avg):
            self.W_h5 = nn.Conv2d(in_channels=self.v_head_out_dim*self.fc_inner_layers_width_mult, out_channels=self.first_linear_out_dim*self.fc_inner_layers_width_mult, kernel_size=1, groups=1, bias=True)
            self.W_h6 = nn.Conv2d(in_channels=self.first_linear_out_dim*self.fc_inner_layers_width_mult, out_channels=self.out_dim_locs+self.out_dim_wts, kernel_size=1, groups=1, bias=True)
        if 0:
            # W_h4 has 2 groups 1 for locations asd one for baseline
            self.W_h4_1 = nn.Conv2d(in_channels=self.W_h4_output_dim * self.nof_sub_networks, out_channels= self.W_h4_output_dim*self.nof_sub_networks, kernel_size=1, groups=self.nof_sub_networks, bias=True)
            self.W_h4_2 = nn.Conv2d(in_channels=self.W_h4_output_dim * self.nof_sub_networks, out_channels= self.W_h4_output_dim*self.nof_sub_networks, kernel_size=1, groups=self.nof_sub_networks, bias=True)


        if not self.skip:
            assert self.particle_embed_dim % self.q_head_in_dim == 0, "d_model % should be zero."
            # print("q_head_in_dim: "+str(self.q_head_in_dim)+", k_head_in_dim: "+str(self.k_head_in_dim)+", v_head_in_dim: "+str(self.v_head_in_dim)+", : "+str(self.)+", particle_embed_dim: "+str(self.particle_embed_dim))
        # assert self.k_head_in_dim>=3
        self.activation_wts = torch.nn.LeakyReLU()

    def get_skip(self):
        return self.skip

    def set_skip_as(self,is_skip):
        self.skip = is_skip

    def forward(self, x0_0):
        #self.count_fpms(x0_0)

        #for param in self.parameters():
        #    print(param)
        #    assert torch.all(torch.isnan(param) == False)
        batch_size, nof_parts, in_dim = x0_0.shape
        nof_targs = int(in_dim/self.state_dim)
        assert torch.allclose(x0_0[:, :, -1], x0_0[:, :, -1] - torch.unsqueeze(torch.max(x0_0[:, :, -1], dim=-1).values, -1) - 1)
        if self.skip:
            return x0_0, None
        else:
            assert self.state_dim*int(in_dim/self.state_dim) != in_dim
            x0 = torch.clone(x0_0)
            # x1,y1,x2,y2,..,w
            assert torch.all(x0[:, :, -1]!=0)
            if self.divide_by_wts:
                x0[:, :, -1] = 1/x0[:, :, -1]
            nof_targs = int(x0.shape[-1]/self.state_dim)
            x1 = torch.cat((torch.reshape(x0[:, :, :-1],(batch_size, nof_parts, nof_targs, self.state_dim)), torch.unsqueeze(torch.tile(x0[:, :, -1:], (1,1,nof_targs)),-1)),dim=3)
            if self.add_nof_targs:
                x1 = torch.cat((x1, torch.full((batch_size, nof_parts, nof_targs, 1), nof_targs, dtype=x1.dtype, device=x1.device, requires_grad=False),
                                torch.full((batch_size, nof_parts, nof_targs, 1), 1/nof_targs, dtype=x1.dtype, device=x1.device, requires_grad=False)), dim=3)
            x1 = torch.permute(x1, (0, 2, 1, 3))
            x1 = torch.reshape(x1,(batch_size*nof_targs*nof_parts, self.embed_in_dim))
            x1 = torch.unsqueeze(torch.unsqueeze(x1,-1),-1)

        x_e16 = self.particle_emb16(x1)
        # batch*targ*particle        |4x(emb_dim)|1|1
        x_e16 = torch.reshape(x_e16, (batch_size, nof_targs, nof_parts, self.sub_n * self.sup_n, self.part_emb_dim))
        if self.sup_n>1:
            x_e_sat = x_e16[:, :,:, :self.sub_n]
            if nof_targs > 1:
                x_e_sat = torch.unsqueeze(torch.sum(x_e_sat, dim=1), 1) - x_e_sat
                x_e_sat = x_e_sat/(nof_targs-1)
            else:
                x_e_sat = torch.zeros_like(x_e_sat)
            x_e16 = x_e16[:, :, :, self.sub_n:] + x_e_sat
        x_e16 = torch.reshape(x_e16, (batch_size*nof_targs, nof_parts, (self.sub_n*self.particle_embed_dim)))

        x = self.sdp16_1(x_e16)
        if self.another_sdp:
            x = self.W_h2(x)#.reshape(batch_size, self.nof_parts, self.nof_sup_networks*self.first_linear_out_dim)
            x = torch.reshape(x, (batch_size* nof_targs, nof_parts, self.nof_sub_networks*self.particle_embed_dim ))
            x = self.sdp16_2(x)
            '''
            end of SDP after dot sum:
            x = self.W_h1(x_xe)  # (bs, q_length, dim)
            x = self.activation_wts(x)
            x = self.activation_wts(self.W_h1_0(x))
            if self.another_layer:
                x = self.activation_wts(self.W_h1_1(x))
                if self.another_another_layer:
                    x = self.activation_wts(self.W_h1_2(x))
            '''
        x1 = self.activation_wts(self.W_h3(x[:, :self.W_h3_W_h4_groups*self.first_linear_out_dim*self.fc_inner_layers_width_mult]))
        x1 = self.activation_wts(self.W_h4(x1))
        x1 = torch.reshape(x1,(batch_size, nof_targs, nof_parts, self.W_h3_W_h4_groups*(self.out_dim_locs+self.out_dim_wts)))
        all_ts_avg_prts_locs_delta = 0
        x3 = None
        if self.additional_flock_for_avg:
            x3 = x1[:, :, :, (self.out_dim_locs + self.out_dim_wts):2*(self.out_dim_locs + self.out_dim_wts):]
            x3 = torch.permute(x3, (0, 2, 1, 3))
            delta_locs = torch.reshape(x3[:, :, :, :self.out_dim_locs], (batch_size, nof_parts, nof_targs * self.out_dim_locs))
            delta_wts = torch.reshape(torch.mean(x3[:, :, :, self.out_dim_locs:], dim=2), (batch_size, nof_parts, self.out_dim_wts))
            x3 = torch.cat((delta_locs, delta_wts), dim=2)
            x3 = x0_0 + x3

        if self.add_baseline_locs:
            if not self.baseline_is_wtd_avg: # W_h5 and W_h5 work on average embedding last third of x (original as in SA1_016_v3)
                x2 = torch.mean(torch.reshape(x[:, self.W_h3_W_h4_groups*self.first_linear_out_dim*self.fc_inner_layers_width_mult:], (batch_size, nof_targs, nof_parts, self.particle_embed_dim*self.fc_inner_layers_width_mult)), dim=2)
                x2 = torch.reshape(x2, (batch_size*nof_targs,self.particle_embed_dim*self.fc_inner_layers_width_mult, 1, 1))
                x2 = self.activation_wts(self.W_h5(x2))
                x2 = self.activation_wts(self.W_h6(x2))
                x2 = torch.reshape(x2, (batch_size, nof_targs, 1, self.out_dim_locs + self.out_dim_wts))
                # first third is particles
                x1 = x1[:, :, :, :self.out_dim_locs + self.out_dim_wts] + x2
            else: # last third is baseline
                x2 = x1[:, :, :, -(self.out_dim_locs + self.out_dim_wts):]
                # batch | targs | particle | dim
                #x2[:, :, :, -1] = x2[:, :, :, -1] - torch.unsqueeze(torch.max(x2[:, :, :, -1], dim=-1).values, -1) - 1
                wts = x2[:, :, :, -1]+torch.unsqueeze(x0_0, dim=1)[:, :,:, -1]
                # batch | targs | particle
                if 0:
                    try:
                        assert torch.sum(wts==0)==0
                    except:
                        print("NN_blocks wts almost nan")
                    #wts2 = torch.where(wts != 0, wts, -torch.finfo().eps)
                    #wts2[0, 0, 0] = -torch.finfo().eps
                    #ddd = 1 / wts2
                    #wts3 = torch.softmax(ddd, dim=-1)
                    wts = torch.where(wts != 0, wts, -torch.finfo().eps)
                    #assert torch.all(wts != 0)
                else:
                    pass
                    #wts = torch.sigmoid(wts) - 1
                #wts = torch.softmax(1 / wts, dim=-1).reshape(batch_size * nof_targs, 1, nof_parts)
                wts = torch.softmax(wts, dim=-1).reshape(batch_size * nof_targs, 1, nof_parts)
                locs_delta = x2[:, :, :, :-1].reshape(batch_size * nof_targs, nof_parts, self.state_dim)
                all_ts_avg_prts_locs_delta = torch.bmm(wts, locs_delta).reshape(batch_size, nof_targs, 1, self.state_dim)
                all_ts_avg_prts_locs_delta = torch.permute(all_ts_avg_prts_locs_delta, (0, 2, 1, 3))

                # batch | targs | 1, state_dim
                # first third is particles
                x1 = x1[:, :, :, :self.out_dim_locs + self.out_dim_wts]
        x = torch.permute(x1, (0, 2, 1, 3))
        delta_locs = torch.reshape(x[:, :, :, :self.out_dim_locs]+all_ts_avg_prts_locs_delta, (batch_size, nof_parts, nof_targs * self.out_dim_locs))
        delta_wts = torch.reshape(torch.mean(x[:, :, :, self.out_dim_locs:], dim=2), (batch_size, nof_parts, self.out_dim_wts))
        x = torch.cat((delta_locs, delta_wts), dim=2)
        if self.divide_by_wts:
            x = x0 + x
        else:
            x = x0_0 + x
        if 1:
            if 1:
                pass
                if self.divide_by_wts:
                    x[:, :, -1] = 1/x[:, :, -1]
        return x, x3

class SA1_016_v24(SA1_016):#same as SA1_016_v21 (single target) with add_baseline_locs=True, fc_inner_layers_width_mult=2, emb_inner_layers_width_mult=2, another_sdp=True
    def __init__(self, skip=False, state_dim=2, particle_embed_dim=2*dim0, q_head_out_dim=2*dim0, v_head_out_dim=2*dim0, add_baseline_locs=True, another_sdp=True):
        # same as SA1_007 but with seperated embeddings for each trgat
        super(SA1_016_v24, self).__init__(skip=skip, state_dim=state_dim, particle_embed_dim=particle_embed_dim,
                                         q_head_out_dim=q_head_out_dim, v_head_out_dim=v_head_out_dim, add_baseline_locs=add_baseline_locs, another_sdp=another_sdp,
                                         emb_inner_layers_width_mult=2, fc_inner_layers_width_mult=2, do_sat_part_emb=False)

class SA1_016_v24MTT(SA1_016):#same as SA1_016_v21 (single target) with add_baseline_locs=True, fc_inner_layers_width_mult=2, emb_inner_layers_width_mult=2, another_sdp=True
    def __init__(self, skip=False, state_dim=2, particle_embed_dim=2*dim0, q_head_out_dim=2*dim0, v_head_out_dim=2*dim0, add_baseline_locs=True, another_sdp=False):
        # same as SA1_007 but with seperated embeddings for each trgat
        super(SA1_016_v24MTT, self).__init__(skip=skip, state_dim=state_dim, particle_embed_dim=particle_embed_dim,
                                         q_head_out_dim=q_head_out_dim, v_head_out_dim=v_head_out_dim, add_baseline_locs=add_baseline_locs, another_sdp=another_sdp,
                                         emb_inner_layers_width_mult=3, fc_inner_layers_width_mult=3, do_sat_part_emb=True)
