import raylibpy as rl
import numpy as np
import random


class Cube:
    def __init__(self, size, center, face_color):
        self.size = size
        self.center = center
        self.face_color = face_color
        self.orientation = np.eye(3)  # initial orientation

        # initialize empty lists for models and update face colors
        self.model = None
        self.gen_meshe(size)
        # create and position the meshes
        self.create_model()

    def gen_meshe(self, scale: tuple):
        # create the central cube meshe
        self.mesh = rl.gen_mesh_cube(*scale)

    # create 3d model
    def create_model(self):
        self.model = rl.load_model_from_mesh(self.mesh)
        self.model.transform = rl.matrix_translate(self.center[0], self.center[1], self.center[2])

    def rotate(self, axis, theta):
        # create the rotation matrix based on the specified axis
        if axis == 0:
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)]
            ])
        elif axis == 1:
            rotation_matrix = np.array([
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)]
            ])
        elif axis == 2:
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
        else:
            raise ValueError("Invalid axis. Use 'x' , 'y', 'z' ")

        # apply the rotation matrix to update the position
        self.center = rotation_matrix @ self.center
        # update the orirntation
        self.orientation = rotation_matrix @ self.orientation

    def get_rotation_axis_angle(self):
        # calculate the rotation axis and angle from orentation matrix
        angle = np.arccos((np.trace(self.orientation) - 1) / 2)
        if angle == 0:
            axis = np.array([0, 0, 0])
        else:
            rx = self.orientation[2, 1] - self.orientation[1, 2]
            ry = self.orientation[0, 2] - self.orientation[2, 0]
            rz = self.orientation[1, 0] - self.orientation[0, 1]
            axis = np.array([rx, ry, rz]) / (2 * np.sin(angle))
        axis = rl.Vector3(axis[0], axis[1], axis[2])
        return axis, np.degrees(angle)


class Rubik:
    def __init__(self) -> None:
        self.cubes = []

        self.is_rotating = False
        self.rotation_angle = 0
        self.rotation_axis = None
        self.level = None
        self.segement = None
        self.target_rotation = 0

        self.generate_rubik(2)

    def generate_rubik(self, size):
        colors = [rl.WHITE, rl.YELLOW, rl.ORANGE, rl.RED, rl.GREEN, rl.BLUE]
        offset = size - 0.7
        size_z = size * 0.9, size * 0.9, size * 0.1
        size_x = size * 0.9, size * 0.1, size * 0.9
        size_y = size * 0.1, size * 0.9, size * 0.9

        for x in range(3):
            for y in range(3):
                for z in range(3):
                    face_colors = [
                        rl.BLACK if z != 2 else colors[0],  # front
                        rl.BLACK if z != 0 else colors[1],  # back
                        rl.BLACK if x != 2 else colors[2],  # right
                        rl.BLACK if x != 0 else colors[3],  # left
                        rl.BLACK if y != 2 else colors[4],  # top
                        rl.BLACK if y != 0 else colors[5]  # bottom
                    ]

                    # center
                    center_position = np.array([
                        (x - 1) * offset,
                        (y - 1) * offset,
                        (z - 1) * offset
                    ])
                    center = Cube((size, size, size), center_position, rl.BLACK)

                    # front_face
                    front_position = np.array([
                        center_position[0],
                        center_position[1],
                        center_position[2] + size / 2
                    ])
                    front = Cube(size_z, front_position, face_colors[0])

                    # back_face
                    back_position = np.array([
                        center_position[0],
                        center_position[1],
                        center_position[2] - size / 2
                    ])
                    back = Cube(size_z, back_position, face_colors[1])
                    # right_face
                    right_position = np.array([
                        center_position[0] + size / 2,
                        center_position[1],
                        center_position[2]
                    ])
                    right = Cube(size_y, right_position, face_colors[2])
                    # left_face
                    left_position = np.array([
                        center_position[0] - size / 2,
                        center_position[1],
                        center_position[2]
                    ])
                    left = Cube(size_y, left_position, face_colors[3])
                    # top_face
                    top_position = np.array([
                        center_position[0],
                        center_position[1] + size / 2,
                        center_position[2]
                    ])
                    top = Cube(size_x, top_position, face_colors[4])
                    # bottom_face
                    bottom_position = np.array([
                        center_position[0],
                        center_position[1] - size / 2,
                        center_position[2]
                    ])
                    bottom = Cube(size_x, bottom_position, face_colors[5])

                    self.cubes.append([center, front, back, right, left, top, bottom])

        return self.cubes

    def choose_piece(self, piece, axis_index, level):
        if level == 0 and round(piece[0].center[axis_index], 1) < 0:
            return True
        elif level == 1 and round(piece[0].center[axis_index], 1) == 0:
            return True
        elif level == 2 and round(piece[0].center[axis_index], 1) > 0:
            return True

        return False

    def get_face(self, axis, level):
        axis_index = np.nonzero(axis)[0][0]
        segement = [i for i, cube in enumerate(self.cubes) \
                    if self.choose_piece(cube, axis_index, level)]
        return segement

    def handle_rotation(self, rotation_queue, animation_step=None):
        # check if there is a request and if not already rotating
        if rotation_queue and not self.is_rotating:
            # get the next rotation axis and level
            self.target_rotation, self.rotation_axis, self.level = rotation_queue.pop(0)
            if self.target_rotation > 0:
                self.target_rotation += random.uniform(0, 1) * 10 ** -3
            else:
                self.target_rotation -= random.uniform(0, 1) * 10 ** -3

            self.segement = self.get_face(self.rotation_axis, self.level)

            # reset rotation angle at the start of a new rotatoin
            self.rotation_angle = 0

            # set rotation to true to start rotating
            self.is_rotating = True

        if self.is_rotating:
            if (self.rotation_angle != self.target_rotation):
                diff = abs(self.target_rotation - self.rotation_angle)
                delta_angle = min(np.radians(1), diff)

                # increment the rotation angle in the correct direction
                self.rotation_angle += delta_angle if self.target_rotation > 0 \
                    else -delta_angle
            else:
                delta_angle = 0

                # stop rotating when target rotation is reached
                self.is_rotating = False
                if animation_step is not None:
                    animation_step += 1
                    # print ('incremented animation step', animation_step)
            
            if self.rotation_axis is not None and isinstance(self.rotation_axis, np.ndarray):
                axis_index = np.nonzero(self.rotation_axis)[0][0]

                for id, cube in enumerate(self.cubes):


                    if id in self.segement:
                        for part_id, _ in enumerate(cube):
                            if self.target_rotation > 0:
                                self.cubes[id][part_id].rotate(axis_index, delta_angle)
                            else:
                                self.cubes[id][part_id].rotate(axis_index, -delta_angle)

                            pos_x, pos_y, pos_z = self.cubes[id][part_id].center

                            translarion = rl.matrix_translate(pos_x, pos_y, pos_z)
                            rota, angle = self.cubes[id][part_id].get_rotation_axis_angle()
                            rotation = rl.matrix_rotate(rota, np.radians(angle))
                            transform = rl.matrix_multiply(rotation, translarion)
                            self.cubes[id][part_id].model.transform = transform

        else:
            self.is_rotating = True

        return rotation_queue, animation_step

    def get_state(self):
        state = []
        # Iterate over each cube and collect the colors of its faces.
        # Use a unique number to represent each color for the state.
        # You might need to create a mapping for your face_colors to numerical ids.
        color_mapping = {
            rl.WHITE.r  * 256**3 + rl.WHITE.g * 256**2 + rl.WHITE.b * 256**1 + rl.WHITE.a: 0,
            rl.YELLOW.r * 256**3 + rl.YELLOW.g * 256**2 + rl.YELLOW.b * 256**1 + rl.YELLOW.a: 1,
            rl.ORANGE.r * 256**3 + rl.ORANGE.g * 256**2 + rl.ORANGE.b * 256**1 + rl.ORANGE.a: 2,
            rl.RED.r * 256**3 + rl.RED.g * 256**2 + rl.RED.b * 256**1 + rl.RED.a: 3,
            rl.GREEN.r * 256**3 + rl.GREEN.g * 256**2 + rl.GREEN.b * 256**1 + rl.GREEN.a: 4,
            rl.BLUE.r * 256**3 + rl.BLUE.g * 256**2 + rl.BLUE.b * 256**1 + rl.BLUE.a: 5,
            rl.BLACK.r * 256**3 + rl.BLACK.g * 256**2 + rl.BLACK.b * 256**1 + rl.BLACK.a: 6  # Represent black as an indicator for 'not relevant'
        }
        for cube in self.cubes:
            for face in cube:
                if face.face_color != rl.BLACK: # exclude the center cube
                    state.append(color_mapping[face.face_color.r  * 256**3 + face.face_color.g * 256**2 + face.face_color.b * 256**1 + face.face_color.a])
        return np.array(state)


    def is_solved(self):
        # the rubik is solved if every face has only one color (excluding the black face).
        for id, cube in enumerate(self.cubes):
            if (id > 25 or id < 3): # only check the corner cubies
                continue
            
            color = None
            for part_id, face in enumerate(cube):
                if part_id == 0: continue # avoid checking the center color
                if face.face_color == rl.BLACK: continue # avoid the black ones
                if color == None:
                    color = face.face_color
                elif color != face.face_color:
                    return False
        return True