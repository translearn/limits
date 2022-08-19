__version__ = '1.1.3'
VERSION = __version__
__version_info__ = tuple([int(sub_version) for sub_version in __version__.split('.')])

from _klimits import normalize
from _klimits import normalize_np
from _klimits import normalize_parallel
from _klimits import normalize_batch
from _klimits import normalize_batch_np
from _klimits import normalize_batch_parallel
from _klimits import denormalize
from _klimits import denormalize_np
from _klimits import denormalize_parallel
from _klimits import calculate_end_position
from _klimits import calculate_end_position_np
from _klimits import calculate_end_position_parallel
from _klimits import interpolate_position
from _klimits import interpolate_position_np
from _klimits import interpolate_position_parallel
from _klimits import interpolate_position_batch
from _klimits import interpolate_position_batch_np
from _klimits import interpolate_position_batch_parallel
from _klimits import calculate_end_velocity
from _klimits import calculate_end_velocity_np
from _klimits import calculate_end_velocity_parallel
from _klimits import interpolate_velocity
from _klimits import interpolate_velocity_np
from _klimits import interpolate_velocity_parallel
from _klimits import interpolate_velocity_batch
from _klimits import interpolate_velocity_batch_np
from _klimits import interpolate_velocity_batch_parallel
from _klimits import interpolate_acceleration
from _klimits import interpolate_acceleration_np
from _klimits import interpolate_acceleration_parallel
from _klimits import interpolate_acceleration_batch
from _klimits import interpolate_acceleration_batch_np
from _klimits import interpolate_acceleration_batch_parallel
from _klimits import compute_distance_c
from _klimits import get_num_threads

from _klimits import PosVelJerkLimitation
