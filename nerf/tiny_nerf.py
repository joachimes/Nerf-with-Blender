import os
from pickle import APPEND
from typing import Optional
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm, trange

#from nerf import cumprod_exclusive, get_minibatches, get_ray_bundle, positional_encoding

def get_minibatches(inputs: torch.Tensor, chunksize: Optional[int] = 1024 * 8):
    r"""Takes a huge tensor (ray "bundle") and splits it into a list of minibatches.
    Each element of the list (except possibly the last) has dimension `0` of length
    `chunksize`.
    """
    return [inputs[i : i + chunksize] for i in range(0, inputs.shape[0], chunksize)]



def cumprod_exclusive(tensor: torch.Tensor) -> torch.Tensor:
    r"""Mimick functionality of tf.math.cumprod(..., exclusive=True), as it isn't available in PyTorch.
    Args:
    tensor (torch.Tensor): Tensor whose cumprod (cumulative product, see `torch.cumprod`) along dim=-1
      is to be computed.
    Returns:
    cumprod (torch.Tensor): cumprod of Tensor along dim=-1, mimiciking the functionality of
      tf.math.cumprod(..., exclusive=True) (see `tf.math.cumprod` for details).
    """
    # TESTED
    # Only works for the last dimension (dim=-1)
    dim = -1
    # Compute regular cumprod first (this is equivalent to `tf.math.cumprod(..., exclusive=False)`).
    cumprod = torch.cumprod(tensor, dim)
    # "Roll" the elements along dimension 'dim' by 1 element.
    cumprod = torch.roll(cumprod, 1, dim)
    # Replace the first element by "1" as this is what tf.cumprod(..., exclusive=True) does.
    cumprod[..., 0] = 1.0

    return cumprod

def meshgrid_xy(
    tensor1: torch.Tensor, tensor2: torch.Tensor
) -> (torch.Tensor, torch.Tensor):
    """Mimick np.meshgrid(..., indexing="xy") in pytorch. torch.meshgrid only allows "ij" indexing.
    (If you're unsure what this means, safely skip trying to understand this, and run a tiny example!)
    Args:
      tensor1 (torch.Tensor): Tensor whose elements define the first dimension of the returned meshgrid.
      tensor2 (torch.Tensor): Tensor whose elements define the second dimension of the returned meshgrid.
    """
    # TESTED
    ii, jj = torch.meshgrid(tensor1, tensor2)
    return ii.transpose(-1, -2), jj.transpose(-1, -2)

def get_ray_bundle(
    height: int, width: int, focal_length: float, tform_cam2world: torch.Tensor
):
    r"""Compute the bundle of rays passing through all pixels of an image (one ray per pixel).
    Args:
    height (int): Height of an image (number of pixels).
    width (int): Width of an image (number of pixels).
    focal_length (float or torch.Tensor): Focal length (number of pixels, i.e., calibrated intrinsics).
    tform_cam2world (torch.Tensor): A 6-DoF rigid-body transform (shape: :math:`(4, 4)`) that
      transforms a 3D point from the camera frame to the "world" frame for the current example.
    Returns:
    ray_origins (torch.Tensor): A tensor of shape :math:`(width, height, 3)` denoting the centers of
      each ray. `ray_origins[i][j]` denotes the origin of the ray passing through pixel at
      row index `j` and column index `i`.
      (TODO: double check if explanation of row and col indices convention is right).
    ray_directions (torch.Tensor): A tensor of shape :math:`(width, height, 3)` denoting the
      direction of each ray (a unit vector). `ray_directions[i][j]` denotes the direction of the ray
      passing through the pixel at row index `j` and column index `i`.
      (TODO: double check if explanation of row and col indices convention is right).
    """
    # TESTED
    ii, jj = meshgrid_xy(
        torch.arange(
            width, dtype=tform_cam2world.dtype, device=tform_cam2world.device
        ).to(tform_cam2world),
        torch.arange(
            height, dtype=tform_cam2world.dtype, device=tform_cam2world.device
        ),
    )
    directions = torch.stack(
        [
            (ii - width * 0.5) / focal_length,
            -(jj - height * 0.5) / focal_length,
            -torch.ones_like(ii),
        ],
        dim=-1,
    )
    ray_directions = torch.sum(
        directions[..., None, :] * tform_cam2world[:3, :3], dim=-1
    )
    ray_origins = tform_cam2world[:3, -1].expand(ray_directions.shape)
    return ray_origins, ray_directions


def positional_encoding(
    tensor, num_encoding_functions=6, include_input=True, log_sampling=True
) -> torch.Tensor:
    r"""Apply positional encoding to the input.
    Args:
        tensor (torch.Tensor): Input tensor to be positionally encoded.
        encoding_size (optional, int): Number of encoding functions used to compute
            a positional encoding (default: 6).
        include_input (optional, bool): Whether or not to include the input in the
            positional encoding (default: True).
    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """
    # TESTED
    # Trivially, the input tensor is added to the positional encoding.
    encoding = [tensor] if include_input else []
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)

# ------------------------------------------------------------------------------------------------
def compute_query_points_from_rays(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    near_thresh: float,
    far_thresh: float,
    num_samples: int,
    randomize: Optional[bool] = True,
) -> (torch.Tensor, torch.Tensor):
    r"""Compute query 3D points given the "bundle" of rays. The near_thresh and far_thresh
    variables indicate the bounds within which 3D points are to be sampled.

    Args:
        ray_origins (torch.Tensor): Origin of each ray in the "bundle" as returned by the
          `get_ray_bundle()` method (shape: :math:`(width, height, 3)`).
        ray_directions (torch.Tensor): Direction of each ray in the "bundle" as returned by the
          `get_ray_bundle()` method (shape: :math:`(width, height, 3)`).
        near_thresh (float): The 'near' extent of the bounding volume (i.e., the nearest depth
          coordinate that is of interest/relevance).
        far_thresh (float): The 'far' extent of the bounding volume (i.e., the farthest depth
          coordinate that is of interest/relevance).
        num_samples (int): Number of samples to be drawn along each ray. Samples are drawn
          randomly, whilst trying to ensure "some form of" uniform spacing among them.
        randomize (optional, bool): Whether or not to randomize the sampling of query points.
          By default, this is set to `True`. If disabled (by setting to `False`), we sample
          uniformly spaced points along each ray in the "bundle".

    Returns:
        query_points (torch.Tensor): Query points along each ray
          (shape: :math:`(width, height, num_samples, 3)`).
        depth_values (torch.Tensor): Sampled depth values along each ray
          (shape: :math:`(num_samples)`).
    """
    # TESTED
    # shape: (num_samples)
    depth_values = torch.linspace(near_thresh, far_thresh, num_samples).to(ray_origins)
    if randomize is True:
        # ray_origins: (width, height, 3)
        # noise_shape = (width, height, num_samples)
        noise_shape = list(ray_origins.shape[:-1]) + [num_samples]
        # depth_values: (num_samples)
        depth_values = (
            depth_values
            + torch.rand(noise_shape).to(ray_origins)
            * (far_thresh - near_thresh)
            / num_samples
        )
    # (width, height, num_samples, 3) = (width, height, 1, 3) + (width, height, 1, 3) * (num_samples, 1)
    # query_points:  (width, height, num_samples, 3)
    query_points = (
        ray_origins[..., None, :]
        + ray_directions[..., None, :] * depth_values[..., :, None]
    )
    # TODO: Double-check that `depth_values` returned is of shape `(num_samples)`.
    return query_points, depth_values


def render_volume_density(
    radiance_field: torch.Tensor, ray_origins: torch.Tensor, depth_values: torch.Tensor
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    r"""Differentiably renders a radiance field, given the origin of each ray in the
    "bundle", and the sampled depth values along them.

    Args:
    radiance_field (torch.Tensor): A "field" where, at each query location (X, Y, Z),
      we have an emitted (RGB) color and a volume density (denoted :math:`\sigma` in
      the paper) (shape: :math:`(width, height, num_samples, 4)`).
    ray_origins (torch.Tensor): Origin of each ray in the "bundle" as returned by the
      `get_ray_bundle()` method (shape: :math:`(width, height, 3)`).
    depth_values (torch.Tensor): Sampled depth values along each ray
      (shape: :math:`(num_samples)`).

    Returns:
    rgb_map (torch.Tensor): Rendered RGB image (shape: :math:`(width, height, 3)`).
    depth_map (torch.Tensor): Rendered depth image (shape: :math:`(width, height)`).
    acc_map (torch.Tensor): # TODO: Double-check (I think this is the accumulated
      transmittance map).
    """
    # TESTED
    sigma_a = torch.nn.functional.relu(radiance_field[..., 3])
    rgb = torch.sigmoid(radiance_field[..., :3])
    one_e_10 = torch.tensor([1e10], dtype=ray_origins.dtype, device=ray_origins.device)
    dists = torch.cat(
        (
            depth_values[..., 1:] - depth_values[..., :-1],
            one_e_10.expand(depth_values[..., :1].shape),
        ),
        dim=-1,
    )
    alpha = 1.0 - torch.exp(-sigma_a * dists)
    weights = alpha * cumprod_exclusive(1.0 - alpha + 1e-10)

    rgb_map = (weights[..., None] * rgb).sum(dim=-2)
    depth_map = (weights * depth_values).sum(dim=-1)
    acc_map = weights.sum(-1)

    return rgb_map, depth_map, acc_map


# One iteration of TinyNeRF (forward pass).
def run_one_iter_of_tinynerf(
    height,
    width,
    focal_length,
    tform_cam2world,
    near_thresh,
    far_thresh,
    depth_samples_per_ray,
    encoding_function,
    get_minibatches_function,
    chunksize,
    model,
    encoding_function_args,
):

    # Get the "bundle" of rays through all image pixels.
    ray_origins, ray_directions = get_ray_bundle(
        height, width, focal_length, tform_cam2world
    )

    # Sample query points along each ray
    query_points, depth_values = compute_query_points_from_rays(
        ray_origins, ray_directions, near_thresh, far_thresh, depth_samples_per_ray
    )

    # "Flatten" the query points.
    flattened_query_points = query_points.reshape((-1, 3))

    # Encode the query points (default: positional encoding).
    encoded_points = encoding_function(flattened_query_points, encoding_function_args)

    # Split the encoded points into "chunks", run the model on all chunks, and
    # concatenate the results (to avoid out-of-memory issues).
    batches = get_minibatches_function(encoded_points, chunksize=chunksize)
    predictions = []
    for batch in batches:
        predictions.append(model(batch))
    radiance_field_flattened = torch.cat(predictions, dim=0)

    # "Unflatten" to obtain the radiance field.
    unflattened_shape = list(query_points.shape[:-1]) + [4]
    radiance_field = torch.reshape(radiance_field_flattened, unflattened_shape)

    # Perform differentiable volume rendering to re-synthesize the RGB image.
    rgb_predicted, _, _ = render_volume_density(
        radiance_field, ray_origins, depth_values
    )

    return rgb_predicted


class VeryTinyNerfModel(torch.nn.Module):
    r"""Define a "very tiny" NeRF model comprising three fully connected layers.
    """

    def __init__(self, filter_size=128, num_encoding_functions=6):
        super(VeryTinyNerfModel, self).__init__()
        # Input layer (default: 39 -> 128)
        self.in_size = 3 + 3 * 2 * num_encoding_functions
        self.layer1 = torch.nn.Linear(self.in_size, filter_size)
        # Layer 2 (default: 128 -> 128)
        self.layer2 = torch.nn.Linear(filter_size, filter_size)
        self.layer3 = torch.nn.Linear(filter_size, filter_size)
        self.layer5 = torch.nn.Linear(filter_size, filter_size)
        self.layer4 = torch.nn.Linear(filter_size + self.in_size, filter_size)
        # Output layer (default: 128 -> 4)
        self.outLayer = torch.nn.Linear(filter_size, 4)
        # Short hand for torch.nn.functional.relu
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        x_in = x.clone()
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.relu(self.layer5(x))
        x = self.relu(torch.cat((x, x_in), -1))
        x = self.relu(self.layer4(x))
        x = self.outLayer(x)
        return x

class TinyNerfModel(torch.nn.Module):
    
    def __init__(self, depth=8, width=128, len_embed=6, skip_connect_every=4):
        super(TinyNerfModel, self).__init__()
        self.depth = depth
        self.width = width
        self.input_size = 3 + 3 * 2 * len_embed
        self.input_linear_layer = torch.nn.Linear(self.input_size, self.width)
        self.skip_connect_every = skip_connect_every
        self.linear_layers = torch.nn.ModuleList()
        for i in range(self.depth):
            if i % self.skip_connect_every == 0 and i > 0 and i != self.depth - 1:
                self.linear_layers.append(
                    torch.nn.Linear(self.input_size + self.width, self.width)
                )
            else:
                self.linear_layers.append(torch.nn.Linear(self.width, self.width))
        # # If not using residual connections.
        # self.linear_layers = torch.nn.ModuleList([
        #     torch.nn.Linear(self.width, self.width)
        # for i in range(depth - 2)])  # depth-2 => account for i/p layer and o/p layer
        self.output_linear_layer = torch.nn.Linear(self.width, 4)
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        x_in = x.clone()
        x = self.input_linear_layer(x)
        for i in range(len(self.linear_layers)):
            if i % self.skip_connect_every == 0 and i > 0 and i != len(self.linear_layers) - 1:
                x = self.relu(torch.cat((x, x_in), -1))
            x = self.relu(self.linear_layers[i](x))
        x = self.output_linear_layer(x)
        return x



def main(data,model, hold_out_img_num=102, num_iters=20000,version=128,near=6,far=12, num_enc=10, depth_samples=32, chunk=4048, save_model=False):

    # Determine device to run on (GPU vs CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Log directory
    logdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "log")
    os.makedirs(logdir, exist_ok=True)

    """
    Load input images and poses
    """

    # Images
    images = data["images"]

    # Camera extrinsics (poses)
    # tform_cam2world = poses 
    tform_cam2world = data["poses"]
    tform_cam2world = torch.from_numpy(np.delete(tform_cam2world, hold_out_img_num, axis=0)).float().to(device)
    # Focal length (intrinsics)
    # focal_length = np.array([55])
    focal_length = data["focal"]
    focal_length = torch.from_numpy(focal_length).float().to(device)
    # Height and width of each image
    height, width = images.shape[1:3]
    # Near and far clipping thresholds for depth values.
    near_thresh = near
    far_thresh = far

    # hold_out_img_num = 102#266
    # Hold one image out (for test).
    testimg, testpose = images[hold_out_img_num], tform_cam2world[hold_out_img_num]
    testimg = torch.from_numpy(testimg).float().to(device)

    # Map images to device
    images = torch.from_numpy(np.delete(images, hold_out_img_num, axis=0)).float().to(device)
    """
    Parameters for TinyNeRF training
    """


    # Specify encoding function.
    encode = positional_encoding
    # Number of depth samples along each ray.
    depth_samples_per_ray = depth_samples

    # Chunksize (Note: this isn't batchsize in the conventional sense. This only
    # specifies the number of rays to be queried in one go. Backprop still happens
    # only after all rays from the current "bundle" are queried and rendered).
    # Use chunksize of about 4096 to fit in ~1.4 GB of GPU memory (when using 8
    # samples per ray).
    chunksize = chunk

    # Optimizer parameters
    lr = 5e-3

    # Misc parameters
    display_every = 100  # Number of iters after which stats are

    """
    Model
    """    
    model.to(device)

    """
    Optimizer
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    """
    Train-Eval-Repeat!
    """

    # Seed RNG, for repeatability
    seed = 9458
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Lists to log metrics etc.
    psnrs = []
    iternums = []

    for i in trange(num_iters):

        # Randomly pick an image as the target.
        target_img_idx = np.random.randint(images.shape[0])
        target_img = images[target_img_idx].to(device)
        target_tform_cam2world = tform_cam2world[target_img_idx].to(device)

        # Run one iteration of TinyNeRF and get the rendered RGB image.
        rgb_predicted = run_one_iter_of_tinynerf(
            height,
            width,
            focal_length,
            target_tform_cam2world,
            near_thresh,
            far_thresh,
            depth_samples_per_ray,
            encode,
            get_minibatches,
            chunksize,
            model,
            num_enc,
        )

        # Compute mean-squared error between the predicted and target images. Backprop!
        loss = torch.nn.functional.mse_loss(rgb_predicted, target_img)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Display images/plots/stats
        if i % display_every == 0 or i == num_iters - 1:
            # Render the held-out view
            rgb_predicted = run_one_iter_of_tinynerf(
                height,
                width,
                focal_length,
                testpose,
                near_thresh,
                far_thresh,
                depth_samples_per_ray,
                encode,
                get_minibatches,
                chunksize,
                model,
                num_enc,
            )
            loss = torch.nn.functional.mse_loss(rgb_predicted, target_img)
            tqdm.write("Loss: " + str(loss.item()))
            psnr = -10.0 * torch.log10(loss)

            psnrs.append(psnr.item())
            iternums.append(i)

            plt.imshow(rgb_predicted.detach().cpu().numpy())
            plt.savefig(os.path.join(logdir, str(i).zfill(6) + ".png"))
            plt.close("all")

            if i == num_iters - 1:
                plt.plot(iternums, psnrs)
                plt.savefig(os.path.join(logdir, "psnr.png"))
                plt.close("all")
            # plt.figure(figsize=(10, 4))
            # plt.subplot(121)
            # plt.imshow(rgb_predicted.detach().cpu().numpy())
            # plt.title(f"Iteration {i}")
            # plt.subplot(122)
            # plt.plot(iternums, psnrs)
            # plt.title("PSNR")
            # plt.show()



    if save_model:
        torch.save(model.state_dict(), f'./weights/{model.__class__.__name__}_weights{version}_{num_iters}_{near}_{far}_{num_enc}.pth')
    else:
        frames = []
        for target_img_idx in range(91, 103):
            # target_img_idx = np.random.randint(images.shape[0]
            target_tform_cam2world = tform_cam2world[target_img_idx].to(device)
            rgb_predicted = run_one_iter_of_tinynerf(
                height,
                width,
                focal_length,
                target_tform_cam2world,
                near_thresh,
                far_thresh,
                depth_samples_per_ray,
                encode,
                get_minibatches,
                chunksize,
                model,
                num_encoding_functions,
            )
            frames.append(Image.fromarray((rgb_predicted.detach().cpu().numpy()*255).astype(np.uint8)))
        frame = frames[0]
        frame.save(os.path.join('./cache/inference', str(target_img_idx).zfill(3) + ".gif"), format='GIF', append_images=frames, save_all=True, duration=100, loop=0)
    print("Done!")

def makeImageNPZ(path, version, num_img=141, file_extension='.png'):
    final_npz = np.zeros((num_img, version, version, 3))
    for i, image in enumerate(os.listdir(os.path.join(path, str(version)))):
        print(i, image, image[:-4])
        if image[-4:] == file_extension:
            final_npz[int(image[:-4])] = plt.imread(os.path.join(path,str(version),image))[:,:,:3] # Ignore last dim if png
    with open(os.path.join(path, f'scene_{version}.npz'), 'wb') as f:
        np.save(f, final_npz)

def do_inference(path_to_weights, model, data, near=8, far=12, num_enc=10, depth_samples=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(path_to_weights))
    model.eval()
    model.to(device)

    # Log directory
    logdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "log")
    os.makedirs(logdir, exist_ok=True)

    """
    Load input images and poses
    """

    # Images
    images = data["images"]

    # Camera extrinsics (poses)
    # tform_cam2world = poses 
    tform_cam2world = data["poses"]
    # Focal length (intrinsics)
    # focal_length = np.array([55])
    focal_length = data["focal"]
    focal_length = torch.from_numpy(focal_length).float().to(device)
    # Height and width of each image
    height, width = images.shape[1:3]
    # Near and far clipping thresholds for depth values.
    near_thresh = near
    far_thresh = far

    # hold_out_img_num = 102#266
    # Hold one image out (for test).

    # Map images to device
    # images = torch.from_numpy(images).float().to(device)
    """
    Parameters for TinyNeRF training
    """

    # Number of functions used in the positional encoding (Be sure to update the
    # model if this number changes).
    num_encoding_functions = num_enc
    # Specify encoding function.
    encode = positional_encoding
    # Number of depth samples along each ray.
    depth_samples_per_ray = depth_samples

    # Chunksize (Note: this isn't batchsize in the conventional sense. This only
    # specifies the number of rays to be queried in one go. Backprop still happens
    # only after all rays from the current "bundle" are queried and rendered).
    # Use chunksize of about 4096 to fit in ~1.4 GB of GPU memory (when using 8
    # samples per ray).
    chunksize = 4096

    frames = []
    for target_img_idx in range(91, 111):
        # target_img_idx = np.random.randint(images.shape[0])
        target_tform_cam2world = torch.from_numpy(tform_cam2world[target_img_idx]).float().to(device)
        # target_tform_cam2world = tform_cam2world[target_img_idx].float().to(device)
        rgb_predicted = run_one_iter_of_tinynerf(
            height,
            width,
            focal_length,
            target_tform_cam2world,
            near_thresh,
            far_thresh,
            depth_samples_per_ray,
            encode,
            get_minibatches,
            chunksize,
            model,
            num_encoding_functions,
        )
        frames.append(Image.fromarray((rgb_predicted.detach().cpu().numpy()*255).astype(np.uint8)))
        # plt.imshow(rgb_predicted.detach().cpu().numpy())
        # plt.savefig(os.path.join('./cache/inference', str(target_img_idx).zfill(3) + ".png"))
        # plt.close("all")

    frame = frames[0]
    frame.save(os.path.join('./cache/inference', str(target_img_idx).zfill(3) + ".gif"), format='GIF', append_images=frames, save_all=True, duration=100, loop=0)

def cleanup(path):
    for f in os.listdir(path):
        os.remove(os.path.join(path,f))

if __name__ == "__main__":
    # m = TinyNerfModel(depth=8)
    # m.cuda()
    # print(m)
    # print(m(torch.rand(2, 39).cuda()).shape)
    version = 128
    # path_to_data = '../data/'
    # makeImageNPZ(path_to_data, version, num_img=211)
    save_model = True
    num_iters = 40001
    near = 2
    far = 6
    # Number of functions used in the positional encoding (Be sure to update the
    # model if this number changes).
    num_encoding_functions = 6
    depth_samples = 110
    focal = np.array([40.]) 
    chunk_size = 4096*64
    layer_size = 64
    data = {'images': np.load(f"../data/scene_{version}.npz"), 'poses': np.load(f"../data/posEnc_{version}.npz"), 'focal': focal }
    # data = np.load(f"cache/tiny_nerf_data.npz")

    # model = TinyNerfModel(depth=3, len_embed=num_encoding_functions)
    model = VeryTinyNerfModel(filter_size=layer_size,num_encoding_functions=num_encoding_functions)
    # w_path = f'./weights/{model.__class__.__name__}_weights{version}_{20001}_{near}_{far}_{num_encoding_functions}.pth'
    # model.load_state_dict(torch.load(w_path))
    cleanup('./cache/log/')
    main(data=data, model=model, hold_out_img_num=168, num_iters=num_iters, version=version, near=near, far=far, num_enc=num_encoding_functions, depth_samples=depth_samples, chunk=chunk_size, save_model=save_model)
    w_path = f'./weights/{model.__class__.__name__}_weights{version}_{num_iters}_{near}_{far}_{num_encoding_functions}.pth'
    do_inference(path_to_weights=w_path,model=model,data=data, near=near, far=far, num_enc=num_encoding_functions, depth_samples=depth_samples)
