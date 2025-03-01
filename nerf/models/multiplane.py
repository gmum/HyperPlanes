import torch
from nerf.nerf_maml_utils import get_embedder

class RenderNetwork(torch.nn.Module):
    def __init__(self, input_size, dir_count):
        super().__init__()
        self.input_size = 3 * input_size + input_size * 3

        self.layers_main = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
        )

        self.layers_main_2 = torch.nn.Sequential(
            torch.nn.Linear(512 + self.input_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
        )

        self.layers_sigma = torch.nn.Sequential(
            torch.nn.Linear(256 + self.input_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
        )

        self.layers_rgb = torch.nn.Sequential(
            torch.nn.Linear(256 + self.input_size + dir_count, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 3),
        )

    def forward(self, triplane_code, viewdir):
        x = self.layers_main(triplane_code)
        x1 = torch.concat([x, triplane_code], dim=1)

        x = self.layers_main_2(x1)
        xs = torch.concat([x, triplane_code], dim=1)

        sigma = self.layers_sigma(xs)
        x = torch.concat([x, triplane_code, viewdir], dim=1)
        rgb = self.layers_rgb(x)
        return torch.concat([rgb, sigma], dim=1)


class ImagePlane(torch.nn.Module):
    def __init__(self, focal, poses, images, count, device="cuda"):
        super(ImagePlane, self).__init__()

        self.pose_matrices = []
        self.K_matrices = []
        self.images = []
        self.centroids = []

        self.focal = focal
        for i in range(min(count, poses.shape[0])):
            M = poses[i]
            #M = torch.from_numpy(M)
            M = M @ torch.Tensor(
                [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            ).to(M.device)
            self.centroids.append(M[0:3, 3])
            M = torch.inverse(M)
            M = M[0:3]
            self.pose_matrices.append(M)

            image = images[i]
            #image = torch.from_numpy(image)
            self.images.append(image.permute(2, 0, 1))
            self.size = float(image.shape[0])
            K = torch.Tensor(
                [
                    [self.focal, 0, 0.5 * image.shape[0]],
                    [0, self.focal, 0.5 * image.shape[0]],
                    [0, 0, 1],
                ]
            ).to(device)

            self.K_matrices.append(K)

        self.pose_matrices = torch.stack(self.pose_matrices)
        self.K_matrices = torch.stack(self.K_matrices)
        self.image_plane = torch.stack(self.images)
        self.centroids = torch.stack(self.centroids)
    def forward(self, points=None):
        if points.shape[0] == 1:
            points = points[0]

        use_xyz = False 
        if use_xyz:
            embed_fn, input_ch = get_embedder(10, 0)
            points = embed_fn(points)
            embed_fn, input_ch = get_embedder(4, 0)
            pose_mat = embed_fn(self.pose_matrices)
            points_shape = points.shape[-1]
            pose_shape = pose_mat.shape[-1]
            pose_mat = torch.cat((pose_mat,
                                torch.ones(size = (25, 3, points_shape-pose_shape)).to(device)), dim = 2)
            ps = self.K_matrices @ pose_mat @ points.T
        else:
            points = torch.concat(
                [points, torch.ones(points.shape[0], 1).to(points.device)], 1
            ).to(points.device) 
            ps = self.K_matrices @ self.pose_matrices @ points.T
            
        pixels = (ps / ps[:, None, 2])[:, 0:2, :]
        pixels = pixels / self.size
        pixels = torch.clamp(pixels, 0, 1)
        pixels = pixels * 2.0 - 1.0
        pixels = pixels.permute(0, 2, 1)

        feats = []
        for img in range(self.image_plane.shape[0]):
            feat = torch.nn.functional.grid_sample(
                self.image_plane[img].unsqueeze(0),
                pixels[img].unsqueeze(0).unsqueeze(0),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            feats.append(feat)
        feats = torch.stack(feats).squeeze(1)
        #feats = torch.mean(feats, dim = 0).unsqueeze(0)
        pixels = pixels.permute(1, 0, 2)
        pixels = pixels.flatten(1)
        feats = feats.permute(2, 3, 0, 1)
        feats = feats.flatten(2)

        cposes = self.centroids.flatten()
        feats = feats[0]
        feats = torch.cat((feats, cposes.unsqueeze(0).repeat(feats.shape[0], 1)), dim=1)
        return feats
    

def generate_planes():
    """
    Defines planes by the three vectors that form the "axes" of the
    plane. Should work with arbitrary number of planes and planes of
    arbitrary orientation.
    """
    
    identity=torch.tensor([[[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]],
                            [[1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0]],
                            [[0, 0, 1],
                            [1, 0, 0],
                            [0, 1, 0]]], dtype=torch.float32)

    # identity=torch.tensor([[[1, 0, 0, 0],
    #                         [0, 1, 0, 0],
    #                         [0, 0, 1, 0],
    #                         [0, 0, 0, 1]],
                           
    #                         [[1, 0, 0, 0],
    #                         [0, 1, 0, 0],
    #                         [0, 0, 1, 0],
    #                         [0, 0, 0, 1]],
                            
    #                         [[1, 0, 0, 0],
    #                         [0, 1, 0, 0],
    #                         [0, 0, 1, 0],
    #                         [0, 0, 0, 1]],
                            
    #                         [[1, 0, 0, 0],
    #                         [0, 1, 0, 0],
    #                         [0, 0, 1, 0],
    #                         [0, 0, 0, 1]]],
                          
                          
    #                       dtype=torch.float32)

    return identity
    

def project_onto_planes(planes, coordinates):
    """
    Does a projection of a 3D point onto a batch of 2D planes,
    returning 2D plane coordinates.

    Takes plane axes of shape n_planes, 3, 3
    # Takes coordinates of shape N, M, 3
    # returns projections of shape N*n_planes, M, 2
    """
    N, M, C = coordinates.shape
    n_planes, _, _ = planes.shape
    coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N*n_planes, M, 3)
    inv_planes = torch.linalg.inv(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N*n_planes, 3, 3).cuda()
    projections = torch.bmm(coordinates, inv_planes)
    return projections[..., :2]

def sample_from_planes(plane_axes, plane_features, coordinates, mode='bilinear', padding_mode='zeros', box_warp=None):
    assert padding_mode == 'zeros'
    N, n_planes, C, H, W = plane_features.shape
    _, M, _ = coordinates.shape
    plane_features = plane_features.view(N*n_planes, C, H, W)

    coordinates = (2/box_warp) * coordinates # TODO: add specific box bounds

    projected_coordinates = project_onto_planes(plane_axes, coordinates).unsqueeze(1)
    output_features = torch.nn.functional.grid_sample(plane_features, projected_coordinates.float(), mode=mode, padding_mode=padding_mode, align_corners=False).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
    return output_features


class TriPlane(torch.nn.Module):
    def __init__(self, focal, poses, images, count, device="cuda"):
        super(TriPlane, self).__init__()

        self.plane_axes = generate_planes()

        self.focal = focal
        self.poses = poses
        
        self.count = count
        self.images = images.view(len(images), 3, 32, images.shape[-2], images.shape[-1])

    def forward(self, points=None):
        points = torch.stack([points for _ in range(self.images.shape[0])], dim=0)
        sampled_features = sample_from_planes(self.plane_axes, self.images, points, padding_mode='zeros', box_warp=1.6)
        return sampled_features



class MultiImageNeRF(torch.nn.Module):
    def __init__(self, image_plane, count, dir_count):
        super(MultiImageNeRF, self).__init__()
        self.image_plane = image_plane
        self.render_network = RenderNetwork(count, dir_count)

        self.input_ch_views = dir_count

    def parameters(self):
        return self.render_network.parameters()

    def set_image_plane(self, ip):
        self.image_plane = ip

    def forward(self, x):
        input_pts, input_views = torch.split(x, [3, self.input_ch_views], dim=-1)
        x = self.image_plane(input_pts)
        return self.render_network(x, input_views)
