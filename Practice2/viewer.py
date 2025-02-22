#!/usr/bin/env python3
"""
Python OpenGL practical application.
"""
# Python built-in modules
import os                           # os function, i.e. checking file status

# External, non built-in modules
import OpenGL.GL as GL              # standard Python OpenGL wrapper
import glfw                         # lean window system wrapper for OpenGL
import numpy as np                  # all matrix manipulations & OpenGL args
import pyassimp                     # 3D ressource loader
import pyassimp.errors              # assimp error management + exceptions

# Own code
from transform import translate, rotate, scale, vec, perspective, lookat

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


# ------------  Simple color shaders ------------------------------------------
COLOR_VERT = """#version 330 core
uniform mat4 matrix;
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
out vec4 outPosition;
void main() {
    outPosition = vec4(color, 1);
    gl_Position = matrix * vec4(position, 1);

}"""

COLOR_FRAG = """#version 330 core
in vec4 outPosition;
out vec4 outColor;
void main() {
    outColor = outPosition;

}"""
# outColor = vec4(color, 1);

# ------------  Scene object classes ------------------------------------------
class VertexArray:
    """ helper class to create and self destroy OpenGL vertex array objects. """
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

class ColorMesh:

    def __init__(self, attributes, index=None):
        self.vertex_array = VertexArray(attributes, index)

    def draw(self, projection, view, model, color_shader):
       GL.glUseProgram(color_shader.glid)
       self.vertex_array.execute(GL.GL_TRIANGLES) 
    
class SimpleTriangle(ColorMesh):
    """Hello triangle object"""

    def __init__(self):
        # triangle position and color buffers
        position = np.array(((0, .5, 0), (.5, -.5, 0), (-.5, -.5, 0)), 'f')
        color = np.array(((1,0,0), (0,1,0), (0,0,1)), 'f')
        super().__init__([position, color])

    def draw(self, projection, view, model, color_shader, color):
        super().draw(projection, view, model, color_shader)

        # model, projection and view transform
        model =  np.identity(4) # translate(0.4, 0.7, 0) @ rotate(vec(1, 0, 0), 25) @ scale(0.7)
        view = lookat(np.array((4, 3, 3), 'f'), np.array((0, 0, 0), 'f'), np.array((0, 1, 0), 'f'))
        projection = perspective(45.0, 4/3, 0.1, 100.0)

        # transformation
        mvp = projection @ view @ model
        matrix_location = GL.glGetUniformLocation(color_shader.glid, 'matrix')
        GL.glUniformMatrix4fv(matrix_location, 1, True, mvp)

class Pyramid(ColorMesh):
    """Pyramid object"""

    def __init__(self):

        # triangle position and color buffers
        position = np.array(((-.5, 0, -.5), (.5, 0, -.5), (.5, 0, .5), (-.5, 0, .5), (0, 1, 0)), np.float32)
        index = np.array((0, 3, 4, 0, 1, 4, 2, 1, 4, 3, 2, 4), np.uint32)
        color = np.array(((1,0,0), (0,1,0), (0,0,1), (1,1,0), (0,1,1)), 'f')
        super().__init__([position, color], index)

    def draw(self, projection, view, model, color_shader, color):
        super().draw(projection, view, model, color_shader)

        # model, projection and view transform
        model =  np.identity(4) # translate(0.4, 0.7, 0) @ rotate(vec(1, 0, 0), 25) @ scale(0.7)
        view = lookat(np.array((0, 3, 3), 'f'), np.array((0, 0, 0), 'f'), np.array((0, 1, 0), 'f'))
        projection = perspective(45.0, 4/3, 0.1, 100.0)

        # transformation
        mvp = projection @ view @ model
        matrix_location = GL.glGetUniformLocation(color_shader.glid, 'matrix')
        GL.glUniformMatrix4fv(matrix_location, 1, True, mvp)

class Pyramid2(ColorMesh):
    """Pyramid object"""

    def __init__(self):

        # triangle position and color buffers
        position = np.array(((-.5, 0, -.5), (.5, 0, -.5), (.5, 0, .5), (-.5, 0, .5), (0, 1, 0)), np.float32)
        position += (2, 0, 0)
        index = np.array((0, 3, 4, 0, 1, 4, 2, 1, 4, 3, 2, 4), np.uint32)
        color = np.array(((1,0,0), (0,1,0), (0,0,1), (1,1,0), (0,1,1)), 'f')
        super().__init__([position, color], index)

    def draw(self, projection, view, model, color_shader, color):
        super().draw(projection, view, model, color_shader)

        # model, projection and view transform
        model =  np.identity(4) # translate(0.4, 0.7, 0) @ rotate(vec(1, 0, 0), 25) @ scale(0.7)
        view = lookat(np.array((0, 3, 3), 'f'), np.array((0, 0, 0), 'f'), np.array((0, 1, 0), 'f'))
        projection = perspective(45.0, 4/3, 0.1, 100.0)

        # transformation
        mvp = projection @ view @ model
        matrix_location = GL.glGetUniformLocation(color_shader.glid, 'matrix')
        GL.glUniformMatrix4fv(matrix_location, 1, True, mvp)

# ------------  Viewer class & window management ------------------------------
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

        # compile and initialize shader programs once globally
        self.color_shader = Shader(COLOR_VERT, COLOR_FRAG)

        # initially empty list of object to draw
        self.drawables = []

        self.color = (0, 1, 0)

    def run(self):
        """ Main render loop for this OpenGL window """
        while not glfw.window_should_close(self.win):
            # clear draw buffer
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)

            # draw our scene objects
            for drawable in self.drawables:
                drawable.draw(None, None, None, self.color_shader)

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
            if key == glfw.KEY_R:
                self.color = (1, 0, 0)
            if key == glfw.KEY_G:
                self.color = (0, 1, 0)
            if key == glfw.KEY_B:
                self.color = (0, 0, 1)

# -------------- 3D ressource loader -----------------------------------------
def load(file):
    """ load resources from file using pyassimp, return list of ColorMesh """
    try:
        print("check 1")
        option = pyassimp.postprocess.aiProcessPreset_TargetRealtime_MaxQuality
        print("check 2")
        scene = pyassimp.load(file, option)
        print("check 3")
    except pyassimp.errors.AssimpError:
        print('ERROR: pyassimp unable to load', file)
        return []     # error reading => return empty list

    meshes = [ColorMesh([m.vertices, m.normals], m.faces) for m in scene.meshes]
    size = sum((mesh.faces.shape[0] for mesh in scene.meshes))
    print('Loaded %s\t(%d meshes, %d faces)' % (file, len(scene.meshes), size))

    pyassimp.release(scene)
    return meshes

# -------------- main program and scene setup --------------------------------
def main():
    """ create a window, add scene objects, then run rendering loop """
    viewer = Viewer()

    # place instances of our basic objects
    #viewer.add(SimpleTriangle())
    #viewer.add(Pyramid())
    #viewer.add(Pyramid2())
    #load("./resources/cylinder.obj")
    for cmesh in load("./resources/suzanne.obj"):
        print("Adding mesh")
        viewer.add(cmesh)
        print("Mesh added")
    # start rendering loop
    viewer.run()


if __name__ == '__main__':
    glfw.init()                # initialize window system glfw
    main()                     # main function keeps variables locally scoped
    glfw.terminate()           # destroy all glfw windows and GL contexts
