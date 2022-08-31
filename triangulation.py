class Triangulation:

    def __init__(self, camera, light):
        self.camera = camera
        self.light = light


    def find_intersection_shadow_plane(self, p):

        # center of camera
        Oc = (self.camera.intrinsic_mat[0][2], self.camera.intrinsic_mat[1][2])

        P0 = self.camera.image_point_to_3d(p)



