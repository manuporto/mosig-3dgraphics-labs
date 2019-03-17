#!/usr/bin/env python3
"""
Python OpenGL practical application.
"""
# Python built-in modules
import os                           # os function, i.e. checking file status
from itertools import cycle
import sys

# External, non built-in modules
import OpenGL.GL as GL              # standard Python OpenGL wrapper
import glfw                         # lean window system wrapper for OpenGL
import numpy as np                  # all matrix manipulations & OpenGL args
import pyassimp                     # 3D resource loader
import pyassimp.errors              # Assimp error management + exceptions

from bisect import bisect_left      # search sorted keyframe lists
from transform import Trackball, identity, translate, rotate, scale, lerp, vec, quaternion_slerp, quaternion_matrix, quaternion, quaternion_from_euler


# ------------ low level OpenGL object wrappers ----------------------------
class Shader:
    """ Helper class to create and automatically destroy shader program """
    @staticmethod
    def _compile_shader(src, shader_type):
        src = open(src, 'r').read() if os.path.exists(src) else src
        src = src.decode('ascii') if isinstance(src, bytes) else src
        shader = GL.glCreateShader(shader_type)
        GL.glShaderSource(shader, src)
        GL.glCompileShader(shader)
        status = GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS)
        src = ('%3d: %s' % (i+1, l) for i, l in enumerate(src.splitlines()))
        if not status:
            log = GL.glGetShaderInfoLog(shader).decode('ascii')
            GL.glDeleteShader(shader)
            src = '\n'.join(src)
            print('Compile failed for %s\n%s\n%s' % (shader_type, log, src))
            return None
        return shader

    def __init__(self, vertex_source, fragment_source):
        """ Shader can be initialized with raw strings or source file names """
        self.glid = None
        vert = self._compile_shader(vertex_source, GL.GL_VERTEX_SHADER)
        frag = self._compile_shader(fragment_source, GL.GL_FRAGMENT_SHADER)
        if vert and frag:
            self.glid = GL.glCreateProgram()  # pylint: disable=E1111
            GL.glAttachShader(self.glid, vert)
            GL.glAttachShader(self.glid, frag)
            GL.glLinkProgram(self.glid)
            GL.glDeleteShader(vert)
            GL.glDeleteShader(frag)
            status = GL.glGetProgramiv(self.glid, GL.GL_LINK_STATUS)
            if not status:
                print(GL.glGetProgramInfoLog(self.glid).decode('ascii'))
                GL.glDeleteProgram(self.glid)
                self.glid = None

    def __del__(self):
        GL.glUseProgram(0)
        if self.glid:                      # if this is a valid shader object
            GL.glDeleteProgram(self.glid)  # object dies => destroy GL object


class VertexArray:
    """ helper class to create and self destroy OpenGL vertex array objects."""
    def __init__(self, attributes, index=None, usage=GL.GL_STATIC_DRAW):
        """ Vertex array from attributes and optional index array. Vertex
            Attributes should be list of arrays with one row per vertex. """

        # create vertex array object, bind it
        self.glid = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.glid)
        self.buffers = []  # we will store buffers in a list
        nb_primitives, size = 0, 0

        # load buffer per vertex attribute (in list with index = shader layout)
        for loc, data in enumerate(attributes):
            if data is not None:
                # bind a new vbo, upload its data to GPU, declare size and type
                self.buffers += [GL.glGenBuffers(1)]
                data = np.array(data, np.float32, copy=False)  # ensure format
                nb_primitives, size = data.shape
                GL.glEnableVertexAttribArray(loc)
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.buffers[-1])
                GL.glBufferData(GL.GL_ARRAY_BUFFER, data, usage)
                GL.glVertexAttribPointer(loc, size, GL.GL_FLOAT, False, 0, None)

        # optionally create and upload an index buffer for this object
        self.draw_command = GL.glDrawArrays
        self.arguments = (0, nb_primitives)
        if index is not None:
            self.buffers += [GL.glGenBuffers(1)]
            index_buffer = np.array(index, np.int32, copy=False)  # good format
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.buffers[-1])
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, index_buffer, usage)
            self.draw_command = GL.glDrawElements
            self.arguments = (index_buffer.size, GL.GL_UNSIGNED_INT, None)

        # cleanup and unbind so no accidental subsequent state update
        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)

    def execute(self, primitive):
        """ draw a vertex array, either as direct array or indexed array """
        GL.glBindVertexArray(self.glid)
        self.draw_command(primitive, *self.arguments)
        GL.glBindVertexArray(0)

    def __del__(self):  # object dies => kill GL array and buffers from GPU
        GL.glDeleteVertexArrays(1, [self.glid])
        GL.glDeleteBuffers(len(self.buffers), self.buffers)


# ------------  simple color fragment shader demonstrated in Practical 1 ------
COLOR_VERT = """#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
out vec3 fragColor;

void main() {
    gl_Position = projection * view * model * vec4(position, 1);
    fragColor = color;
}"""


COLOR_FRAG = """#version 330 core
in vec3 fragColor;
out vec4 outColor;
void main() {
    outColor = vec4(fragColor, 1);
}"""


# ------------  Scene object classes ------------------------------------------

class Node:
    """ Scene graph transform and parameter broadcast node """
    def __init__(self, name='', children=(), transform=identity(), **param):
        self.transform, self.param, self.name = transform, param, name
        self.children = list(iter(children))

    def add(self, *drawables):
        """ Add drawables to this node, simply updating children list """
        self.children.extend(drawables)

    def draw(self, projection, view, model, **param):
        """ Recursive draw, passing down named parameters & model matrix. """
        # merge named parameters given at initialization with those given here
        param = dict(param, **self.param)
        # model = ...   # what to insert here for hierarchical update?
        model = model @ self.transform
        for child in self.children:
            # model = model @ self.transform
            child.draw(projection, view, model, **param)


# mesh to refactor all previous classes
class ColorMesh:

    def __init__(self, attributes, index=None):
        self.vertex_array = VertexArray(attributes, index)

    def draw(self, projection, view, model, color_shader, **param):

        names = ['view', 'projection', 'model']
        loc = {n: GL.glGetUniformLocation(color_shader.glid, n) for n in names}
        GL.glUseProgram(color_shader.glid)

        GL.glUniformMatrix4fv(loc['view'], 1, True, view)
        GL.glUniformMatrix4fv(loc['projection'], 1, True, projection)
        GL.glUniformMatrix4fv(loc['model'], 1, True, model)

        # draw triangle as GL_TRIANGLE vertex array, draw array call
        self.vertex_array.execute(GL.GL_TRIANGLES)


class SimpleTriangle(ColorMesh):
    """Hello triangle object"""

    def __init__(self):

        # triangle position buffer
        position = np.array(((0, .5, 0), (.5, -.5, 0), (-.5, -.5, 0)), 'f')
        color = np.array(((1, 0, 0), (0, 1, 0), (0, 0, 1)), 'f')

        super().__init__([position, color])


class Cylinder(Node):
    """ Very simple cylinder based on practical 2 load function """
    def __init__(self):
        super().__init__()
        self.add(*load('./resources/cylinder.obj'))  # just load the cylinder from file

# -------------- Interpolator class -----------------------------------------
class KeyFrames:
    """ Stores keyframe pairs for any value type with interpolation_function"""
    def __init__(self, time_value_pairs, interpolation_function=lerp):
        if isinstance(time_value_pairs, dict):  # convert to list of pairs
            time_value_pairs = time_value_pairs.items()
        keyframes = sorted(((key[0], key[1]) for key in time_value_pairs))
        self.times, self.values = zip(*keyframes)  # pairs list -> 2 lists
        self.interpolate = interpolation_function

    def value(self, time):
        """ Computes interpolated value from keyframes, for a given time """

        # 1. ensure time is within bounds else return boundary keyframe
        if time <= self.times[0]:
            return self.values[0] # i need to return the values not the time
        if time >= self.times[-1]:
            return self.values[-1]
        # 2. search for closest index entry in self.times, using bisect_left function
        index_entry = bisect_left(self.times, time)
        # 3. using the retrieved index, interpolate between the two neighboring values
        # in self.values, using the initially stored self.interpolate function
        fraction = (time - self.times[index_entry - 1]) / (self.times[index_entry] - self.times[index_entry-1])
        return self.interpolate(self.values[index_entry - 1], self.values[index_entry], fraction)

class TransformKeyFrames:
    """ KeyFrames-like object dedicated to 3D transforms """
    def __init__(self, translate_keys, rotate_keys, scale_keys):
        """ stores 3 keyframe sets for translation, rotation, scale """
        self.translate_keys = KeyFrames(translate_keys)
        self.rotate_keys = KeyFrames(rotate_keys, quaternion_slerp)
        self.scale_keys = KeyFrames(scale_keys)

    def value(self, time):
        """ Compute each component's interpolation and compose TRS matrix """
        translate_mat = translate(self.translate_keys.value(time))
        rotate_mat = quaternion_matrix(self.rotate_keys.value(time))
        scale_mat = scale(self.scale_keys.value(time))
        return translate_mat @ rotate_mat @ scale_mat

class KeyFrameControlNode(Node):
    """ Place node with transform keys above a controlled subtree """
    def __init__(self, translate_keys, rotate_keys, scale_keys, **kwargs):
        super().__init__(**kwargs)
        self.keyframes = TransformKeyFrames(translate_keys, rotate_keys, scale_keys)

    def draw(self, projection, view, model, **param):
        """ When redraw requested, interpolate our node transform from keys """
        self.transform = self.keyframes.value(glfw.get_time())
        super().draw(projection, view, model, **param)

# -------------- 3D ressource loader -----------------------------------------
def load(file):
    """ load resources from file using pyassimp, return list of ColorMesh """
    try:
        option = pyassimp.postprocess.aiProcessPreset_TargetRealtime_MaxQuality
        scene = pyassimp.load(file, option)
    except pyassimp.errors.AssimpError:
        print('ERROR: pyassimp unable to load', file)
        return []  # error reading => return empty list

    meshes = [ColorMesh([m.vertices, m.normals], m.faces) for m in scene.meshes]
    size = sum((mesh.faces.shape[0] for mesh in scene.meshes))
    print('Loaded %s\t(%d meshes, %d faces)' % (file, len(scene.meshes), size))

    pyassimp.release(scene)
    return meshes


# ------------  Viewer class & window management ------------------------------
class GLFWTrackball(Trackball):
    """ Use in Viewer for interactive viewpoint control """

    def __init__(self, win):
        """ Init needs a GLFW window handler 'win' to register callbacks """
        super().__init__()
        self.mouse = (0, 0)
        glfw.set_cursor_pos_callback(win, self.on_mouse_move)
        glfw.set_scroll_callback(win, self.on_scroll)

    def on_mouse_move(self, win, xpos, ypos):
        """ Rotate on left-click & drag, pan on right-click & drag """
        old = self.mouse
        self.mouse = (xpos, glfw.get_window_size(win)[1] - ypos)
        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT):
            self.drag(old, self.mouse, glfw.get_window_size(win))
        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_RIGHT):
            self.pan(old, self.mouse)

    def on_scroll(self, win, _deltax, deltay):
        """ Scroll controls the camera distance to trackball center """
        self.zoom(deltay, glfw.get_window_size(win)[1])


class Viewer:
    """ GLFW viewer window, with classic initialization & graphics loop """

    def __init__(self, width=640, height=480):

        # version hints: create GL window with >= OpenGL 3.3 and core profile
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, False)
        self.win = glfw.create_window(width, height, 'Viewer', None, None)

        # make win's OpenGL context current; no OpenGL calls can happen before
        glfw.make_context_current(self.win)

        # register event handlers
        glfw.set_key_callback(self.win, self.on_key)

        # useful message to check OpenGL renderer characteristics
        print('OpenGL', GL.glGetString(GL.GL_VERSION).decode() + ', GLSL',
              GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION).decode() +
              ', Renderer', GL.glGetString(GL.GL_RENDERER).decode())

        # initialize GL by setting viewport and default render characteristics
        GL.glClearColor(0.1, 0.1, 0.1, 0.1)
        GL.glEnable(GL.GL_DEPTH_TEST)         # depth test now enabled (TP2)
        GL.glEnable(GL.GL_CULL_FACE)          # backface culling enabled (TP2)

        # compile and initialize shader programs once globally
        self.color_shader = Shader(COLOR_VERT, COLOR_FRAG)

        # initially empty list of object to draw
        self.drawables = []

        # initialize trackball
        self.trackball = GLFWTrackball(self.win)

        # cyclic iterator to easily toggle polygon rendering modes
        self.fill_modes = cycle([GL.GL_LINE, GL.GL_POINT, GL.GL_FILL])
        
    def run(self):
        """ Main render loop for this OpenGL window """
        while not glfw.window_should_close(self.win):
            # clear draw buffer and depth buffer (<-TP2)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

            winsize = glfw.get_window_size(self.win)
            view = self.trackball.view_matrix()
            projection = self.trackball.projection_matrix(winsize)
            
            # draw our scene objects
            for drawable in self.drawables:
                drawable.draw(projection, view, identity(),
                              color_shader=self.color_shader, win=self.win)

            # flush render commands, and swap draw buffers
            glfw.swap_buffers(self.win)

            # Poll for and process events
            glfw.poll_events()

    def add(self, *drawables):
        """ add objects to draw in this window """
        self.drawables.extend(drawables)

    def on_key(self, _win, key, _scancode, action, _mods):
        """ 'Q' or 'Escape' quits """
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_ESCAPE or key == glfw.KEY_Q:
                glfw.set_window_should_close(self.win, True)
            if key == glfw.KEY_W:
                GL.glPolygonMode(GL.GL_FRONT_AND_BACK, next(self.fill_modes))


# -------------- main program and scene setup --------------------------------
def main():
    """ create a window, add scene objects, then run rendering loop """
    viewer = Viewer()

    translate_keys = {0: vec(0, 0, 0), 2: vec(1, 1, 0), 4: vec(0, 0, 0)}
    rotate_keys = {0: quaternion(), 2: quaternion_from_euler(180, 45, 90),
                   3: quaternion_from_euler(180, 0, 180), 4: quaternion()}
    scale_keys = {0: 1, 2: 0.5, 4: 1}
    keynode = KeyFrameControlNode(translate_keys, rotate_keys, scale_keys)
    keynode.add(Cylinder())
    viewer.add(keynode)

    # start rendering loop
    viewer.run()


if __name__ == '__main__':
    glfw.init()                # initialize window system glfw
    main()                     # main function keeps variables locally scoped
    glfw.terminate()           # destroy all glfw windows and GL contexts
