import warnings
from clothoids.multiple_sequential_asymmetric_clothoid_spline import create_clothoid_spline_from_control_points
from clothoids.multiple_sequential_asymmetric_clothoid_spline import create_clothoid_spline_from_middle_points

warnings.filterwarnings("ignore", category=RuntimeWarning)


def clothoids_from_control_points(points, tau=0.5, clothoid_point_step=0.01, clothoid_point_search_step=0.01,
                                  approximate_fresnel=False, factors=None):
    """
    Create multiple sequential asymmetric clothoid spline from control points
    Args:
        points (numpy 2-D array): numpy array of 2-D points, e.g., [[5,1],[4,10],[1,20],[6,30]]
        tau: ...
        clothoid_point_step (float): step of clothoid points on each create_clothoids; default = 0.01
        clothoid_point_search_step (float): step for finding the shortest distance on clothoid; default = 0.01
        approximate_fresnel (boolean): if True, the Fresnel integrals are approximated; default = False
        factors (list of floats): factor for each clothoid; default = 0.5
    Returns:
        points (numpy 2-D array)
    """
    [route_center, alpha, norm_dist] = create_clothoid_spline_from_control_points(
        points.tolist(), tau, clothoid_point_step, clothoid_point_search_step, approximate_fresnel,
        factors)
    return route_center


def clothoids_from_middle_points(points, clothoid_point_step=0.01, clothoid_point_search_step=0.01,
                                 approximate_fresnel=False):
    """
    Create multiple sequential asymmetric clothoid spline from middle points and angles
    Args:
        points (numpy 2-D array): numpy array of 2-D points with angles (in radians), e.g.,
        [[5,1,math.pi],[4,10,math.pi/3],[1,20,math.pi/6],[6,30,math.pi/8]]
        tau: ...
        clothoid_point_step (float): step of clothoid points on each create_clothoids; default = 0.01
        clothoid_point_search_step (float): step for finding the shortest distance on clothoid; default = 0.01
        approximate_fresnel (boolean): if True, the Fresnel integrals are approximated; default = False
    Returns:
        points (numpy 2-D array)
    """
    route_center, norm_dist = create_clothoid_spline_from_middle_points(
        points.tolist(), 0.5, clothoid_point_step, clothoid_point_search_step, approximate_fresnel)
    return route_center

