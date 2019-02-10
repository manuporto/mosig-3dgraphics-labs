### How to store position's info about triangles
* Triangle soup
* Indexed array (separate geometry from topology and colors of the triangles)

### Vertex Shader
* Executed for one vertex and done for all the vertexes in parallel
* Matrix multiplication done in the Vertex Shader step (VS). (position, camera and projection matrixes)
* P*(V(M*vec4(position, 4))) (all matrixes are 4x4) (projection, view and model matrixes)

### Fragment Shader
* You are walking on pixels not on vertexes
* At this stage you compute the final color

### How do we send data to the GPU?

### Notes
