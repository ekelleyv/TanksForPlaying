// Source file for the scene file viewer



////////////////////////////////////////////////////////////
// INCLUDE FILES
////////////////////////////////////////////////////////////

#include "R3/R3.h"
#include "R3Scene.h"
#include "cos426_opengl.h"
#include <float.h>
#include <time.h>
#include "IrrKlang/include/irrKlang.h"

////////////////////////////////////////////////////////////
// GLOBAL VARIABLES
////////////////////////////////////////////////////////////

// Program arguments

static char *input_scene_name = NULL;
static char *output_image_name = NULL;



// Display variables

static R3Scene *scene = NULL;
static R3Camera camera;
static R3Camera camera2;
static int show_faces = 1;
static int show_edges = 0;
static int show_bboxes = 0;
static int show_lights = 0;
static int show_camera = 0;
static int save_image = 0;
static int quit = 0;

//Movement Variables
//Key buffering
//http://www.swiftless.com/tutorials/opengl/keyboard.html
bool* key_states = new bool[256];
bool* key_special_states = new bool[256];

//Camera Variables
bool first_person = 1;
bool third_person = 0;
bool first_person_red = 1;
bool third_person_red = 0;
bool turning = false;


// GLUT variables 

static int GLUTwindow = 0;
static int GLUTwindow_height = 720;
static int GLUTwindow_width = 1080;
static int GLUTmouse[2] = { 0, 0 };
static int GLUTbutton[3] = { 0, 0, 0 };
static int GLUTmodifiers = 0;

//irrKlang variables
irrklang::ISoundEngine* engine = irrklang::createIrrKlangDevice();

//Texture
R2Image map_texture_image;
R3Material map_material;

// GLUT command list

enum {
  DISPLAY_FACE_TOGGLE_COMMAND,
  DISPLAY_EDGE_TOGGLE_COMMAND,
  DISPLAY_BBOXES_TOGGLE_COMMAND,
  DISPLAY_LIGHTS_TOGGLE_COMMAND,
  DISPLAY_CAMERA_TOGGLE_COMMAND,
  SAVE_IMAGE_COMMAND,
  QUIT_COMMAND,
};

bool high_quality = 1;
R3Mesh *high_quality_bluetankmesh;
R3Mesh *low_quality_bluetankmesh;
R3Mesh *high_quality_redtankmesh;
R3Mesh *low_quality_redtankmesh;
////////////////////////////////////////////////////////////
// TIMER CODE
////////////////////////////////////////////////////////////

#ifdef _WIN32
#  include <windows.h>
#else
#  include <sys/time.h>
#endif

static double GetTime(void)
{
#ifdef _WIN32
  // Return number of seconds since start of execution
  static int first = 1;
  static LARGE_INTEGER timefreq;
  static LARGE_INTEGER start_timevalue;

  // Check if this is the first time
  if (first) {
    // Initialize first time
    QueryPerformanceFrequency(&timefreq);
    QueryPerformanceCounter(&start_timevalue);
    first = 0;
    return 0;
  }
  else {
    // Return time since start
    LARGE_INTEGER current_timevalue;
    QueryPerformanceCounter(&current_timevalue);
    return ((double) current_timevalue.QuadPart - 
            (double) start_timevalue.QuadPart) / 
            (double) timefreq.QuadPart;
  }
#else
  // Return number of seconds since start of execution
  static int first = 1;
  static struct timeval start_timevalue;

  // Check if this is the first time
  if (first) {
    // Initialize first time
    gettimeofday(&start_timevalue, NULL);
    first = 0;
    return 0;
  }
  else {
    // Return time since start
    struct timeval current_timevalue;
    gettimeofday(&current_timevalue, NULL);
    int secs = current_timevalue.tv_sec - start_timevalue.tv_sec;
    int usecs = current_timevalue.tv_usec - start_timevalue.tv_usec;
    return (double) (secs + 1.0E-6F * usecs);
  }
#endif
}



////////////////////////////////////////////////////////////
// SCENE DRAWING CODE
////////////////////////////////////////////////////////////

void DrawShape(R3Shape *shape)
{
  // Check shape type
  if (shape->type == R3_BOX_SHAPE) shape->box->Draw();
  else if (shape->type == R3_SPHERE_SHAPE) shape->sphere->Draw();
  else if (shape->type == R3_CYLINDER_SHAPE) shape->cylinder->Draw();
  else if (shape->type == R3_CONE_SHAPE) shape->cone->Draw();
  else if (shape->type == R3_MESH_SHAPE) shape->mesh->Draw();
  else if (shape->type == R3_SEGMENT_SHAPE) shape->segment->Draw();
  else fprintf(stderr, "Unrecognized shape type: %d\n", shape->type);
}



void LoadMatrix(R3Matrix *matrix)
{
  // Multiply matrix by top of stack
  // Take transpose of matrix because OpenGL represents vectors with 
  // column-vectors and R3 represents them with row-vectors
  R3Matrix m = matrix->Transpose();
  glMultMatrixd((double *) &m);
}



void LoadMaterial(R3Material *material) 
{
  GLfloat c[4];

  // Check if same as current
  static R3Material *current_material = NULL;
  if (material == current_material) return;
  current_material = material;

  // Compute "opacity"
  double opacity = 1 - material->kt.Luminance();

  // Load ambient
  c[0] = material->ka[0];
  c[1] = material->ka[1];
  c[2] = material->ka[2];
  c[3] = opacity;
  glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, c);

  // Load diffuse
  c[0] = material->kd[0];
  c[1] = material->kd[1];
  c[2] = material->kd[2];
  c[3] = opacity;
  glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, c);

  // Load specular
  c[0] = material->ks[0];
  c[1] = material->ks[1];
  c[2] = material->ks[2];
  c[3] = opacity;
  glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, c);

  // Load emission
  c[0] = material->emission.Red();
  c[1] = material->emission.Green();
  c[2] = material->emission.Blue();
  c[3] = opacity;
  glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, c);

  // Load shininess
  c[0] = material->shininess;
  glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, c[0]);

  // Load texture
  if (high_quality && material->texture) {
    if (material->texture_index <= 0) {
      // Create texture in OpenGL
      GLuint texture_index;
      glGenTextures(1, &texture_index);
      material->texture_index = (int) texture_index;
      glBindTexture(GL_TEXTURE_2D, material->texture_index); 
      R2Image *image = material->texture;
      int npixels = image->NPixels();
      R2Pixel *pixels = image->Pixels();
      GLfloat *buffer = new GLfloat [ 4 * npixels ];
      R2Pixel *pixelsp = pixels;
      GLfloat *bufferp = buffer;
      for (int j = 0; j < npixels; j++) { 
        *(bufferp++) = pixelsp->Red();
        *(bufferp++) = pixelsp->Green();
        *(bufferp++) = pixelsp->Blue();
        *(bufferp++) = pixelsp->Alpha();
        pixelsp++;
      }
      glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_REPEAT);
      glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_REPEAT);
      glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
      glTexImage2D(GL_TEXTURE_2D, 0, 4, image->Width(), image->Height(), 0, GL_RGBA, GL_FLOAT, buffer);
      delete [] buffer;
    }

    // Select texture
    glBindTexture(GL_TEXTURE_2D, material->texture_index); 
    glEnable(GL_TEXTURE_2D);
  }
  else {
    glDisable(GL_TEXTURE_2D);
  }

  // Enable blending for transparent surfaces
  if (opacity < 1) {
    glDepthMask(false);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);
  }
  else {
    glDisable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ZERO);
    glDepthMask(true);
  }
}


void LoadCamera(R3Camera *camera)
{
  // Set projection transformation
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(2*180.0*camera->yfov/M_PI, (GLdouble) GLUTwindow_width/2 /(GLdouble) GLUTwindow_height, 0.01, 10000);

  // Set camera transformation
  R3Vector t = -(camera->towards);
  R3Vector& u = camera->up;
  R3Vector& r = camera->right;
  GLdouble camera_matrix[16] = { r[0], u[0], t[0], 0, r[1], u[1], t[1], 0, r[2], u[2], t[2], 0, 0, 0, 0, 1 };
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glMultMatrixd(camera_matrix);
  glTranslated(-(camera->eye[0]), -(camera->eye[1]), -(camera->eye[2]));
}

void LoadCamera2(R3Camera *camera)
{
  // Set projection transformation
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(2*180.0*camera->yfov/M_PI, (GLdouble) GLUTwindow_width/2 /(GLdouble) GLUTwindow_height, 0.01, 10000);

  // Set camera transformation
  R3Vector t = -(camera->towards);
  R3Vector& u = camera->up;
  R3Vector& r = camera->right;
  GLdouble camera_matrix[16] = { r[0], u[0], t[0], 0, r[1], u[1], t[1], 0, r[2], u[2], t[2], 0, 0, 0, 0, 1 };
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glMultMatrixd(camera_matrix);
  glTranslated(-(camera->eye[0]), -(camera->eye[1]), -(camera->eye[2]));
}



void LoadLights(R3Scene *scene)
{
  GLfloat buffer[4];

  // Load ambient light
  static GLfloat ambient[4];
  ambient[0] = scene->ambient[0];
  ambient[1] = scene->ambient[1];
  ambient[2] = scene->ambient[2];
  ambient[3] = 1;
  glLightModelfv(GL_LIGHT_MODEL_AMBIENT, ambient);

  // Load scene lights
  for (int i = 0; i < (int) scene->lights.size(); i++) {
    R3Light *light = scene->lights[i];
    int index = GL_LIGHT0 + i;

    // Temporarily disable light
    glDisable(index);

    // Load color
    buffer[0] = light->color[0];
    buffer[1] = light->color[1];
    buffer[2] = light->color[2];
    buffer[3] = 1.0;
    glLightfv(index, GL_DIFFUSE, buffer);
    glLightfv(index, GL_SPECULAR, buffer);

    // Load attenuation with distance
    buffer[0] = light->constant_attenuation;
    buffer[1] = light->linear_attenuation;
    buffer[2] = light->quadratic_attenuation;
    glLightf(index, GL_CONSTANT_ATTENUATION, buffer[0]);
    glLightf(index, GL_LINEAR_ATTENUATION, buffer[1]);
    glLightf(index, GL_QUADRATIC_ATTENUATION, buffer[2]);

    // Load spot light behavior
    buffer[0] = 180.0 * light->angle_cutoff / M_PI;
    buffer[1] = light->angle_attenuation;
    glLightf(index, GL_SPOT_CUTOFF, buffer[0]);
    glLightf(index, GL_SPOT_EXPONENT, buffer[1]);

    // Load positions/directions
    if (light->type == R3_DIRECTIONAL_LIGHT) {
      // Load direction
      buffer[0] = -(light->direction.X());
      buffer[1] = -(light->direction.Y());
      buffer[2] = -(light->direction.Z());
      buffer[3] = 0.0;
      glLightfv(index, GL_POSITION, buffer);
    }
    else if (light->type == R3_POINT_LIGHT) {
      // Load position
      buffer[0] = light->position.X();
      buffer[1] = light->position.Y();
      buffer[2] = light->position.Z();
      buffer[3] = 1.0;
      glLightfv(index, GL_POSITION, buffer);
    }
    else if (light->type == R3_SPOT_LIGHT) {
      // Load position
      buffer[0] = light->position.X();
      buffer[1] = light->position.Y();
      buffer[2] = light->position.Z();
      buffer[3] = 1.0;
      glLightfv(index, GL_POSITION, buffer);

      // Load direction
      buffer[0] = light->direction.X();
      buffer[1] = light->direction.Y();
      buffer[2] = light->direction.Z();
      buffer[3] = 1.0;
      glLightfv(index, GL_SPOT_DIRECTION, buffer);
    }
    else if (light->type == R3_AREA_LIGHT) {
      // Load position
      buffer[0] = light->position.X();
      buffer[1] = light->position.Y();
      buffer[2] = light->position.Z();
      buffer[3] = 1.0;
      glLightfv(index, GL_POSITION, buffer);

      // Load direction
      buffer[0] = light->direction.X();
      buffer[1] = light->direction.Y();
      buffer[2] = light->direction.Z();
      buffer[3] = 1.0;
      glLightfv(index, GL_SPOT_DIRECTION, buffer);
    }
     else {
      fprintf(stderr, "Unrecognized light type: %d\n", light->type);
      return;
    }

    // Enable light
    glEnable(index);
  }
}



void DrawNode(R3Scene *scene, R3Node *node)
{
  // Push transformation onto stack
  glPushMatrix();
  LoadMatrix(&node->transformation);

  // Load material
  if (node->material) LoadMaterial(node->material);

  // Draw shape
  if (node->shape) DrawShape(node->shape);

  // Draw children nodes
  for (int i = 0; i < (int) node->children.size(); i++) 
    DrawNode(scene, node->children[i]);

  // Restore previous transformation
  glPopMatrix();

  // Show bounding box
  if (show_bboxes) {
    GLboolean lighting = glIsEnabled(GL_LIGHTING);
    glDisable(GL_LIGHTING);
    node->bbox.Outline();
    if (lighting) glEnable(GL_LIGHTING);
  }
}



void DrawLights(R3Scene *scene)
{
  // Check if should draw lights
  if (!show_lights) return;

  // Setup
  GLboolean lighting = glIsEnabled(GL_LIGHTING);
  glDisable(GL_LIGHTING);

  // Draw all lights
  double radius = scene->bbox.DiagonalRadius();
  for (int i = 0; i < scene->NLights(); i++) {
    R3Light *light = scene->Light(i);
    glColor3d(light->color[0], light->color[1], light->color[2]);
    if (light->type == R3_DIRECTIONAL_LIGHT) {
      // Draw direction vector
      glLineWidth(5);
      glBegin(GL_LINES);
      R3Point centroid = scene->bbox.Centroid();
      R3Vector vector = radius * light->direction;
      glVertex3d(centroid[0] - vector[0], centroid[1] - vector[1], centroid[2] - vector[2]);
      glVertex3d(centroid[0] - 1.25*vector[0], centroid[1] - 1.25*vector[1], centroid[2] - 1.25*vector[2]);
      glEnd();
      glLineWidth(1);
    }
    else if (light->type == R3_POINT_LIGHT) {
      // Draw sphere at point light position
      R3Sphere(light->position, 0.1 * radius).Draw();
    }
    else if (light->type == R3_SPOT_LIGHT) {
      // Draw sphere at point light position and line indicating direction
      R3Sphere(light->position, 0.1 * radius).Draw();
  
      // Draw direction vector
      glLineWidth(5);
      glBegin(GL_LINES);
      R3Vector vector = radius * light->direction;
      glVertex3d(light->position[0], light->position[1], light->position[2]);
      glVertex3d(light->position[0] + 0.25*vector[0], light->position[1] + 0.25*vector[1], light->position[2] + 0.25*vector[2]);
      glEnd();
      glLineWidth(1);
    }
    else if (light->type == R3_AREA_LIGHT) {
      // Draw circular area
      R3Vector v1, v2;
      double r = light->radius;
      R3Point p = light->position;
      int dim = light->direction.MinDimension();
      if (dim == 0) { v1 = light->direction % R3posx_vector; v1.Normalize(); v2 = light->direction % v1; }
      else if (dim == 1) { v1 = light->direction % R3posy_vector; v1.Normalize(); v2 = light->direction % v1; }
      else { v1 = light->direction % R3posz_vector; v1.Normalize(); v2 = light->direction % v1; }
      glBegin(GL_POLYGON);
      glVertex3d(p[0] +  1.00*r*v1[0] +  0.00*r*v2[0], p[1] +  1.00*r*v1[1] +  0.00*r*v2[1], p[2] +  1.00*r*v1[2] +  0.00*r*v2[2]);
      glVertex3d(p[0] +  0.71*r*v1[0] +  0.71*r*v2[0], p[1] +  0.71*r*v1[1] +  0.71*r*v2[1], p[2] +  0.71*r*v1[2] +  0.71*r*v2[2]);
      glVertex3d(p[0] +  0.00*r*v1[0] +  1.00*r*v2[0], p[1] +  0.00*r*v1[1] +  1.00*r*v2[1], p[2] +  0.00*r*v1[2] +  1.00*r*v2[2]);
      glVertex3d(p[0] + -0.71*r*v1[0] +  0.71*r*v2[0], p[1] + -0.71*r*v1[1] +  0.71*r*v2[1], p[2] + -0.71*r*v1[2] +  0.71*r*v2[2]);
      glVertex3d(p[0] + -1.00*r*v1[0] +  0.00*r*v2[0], p[1] + -1.00*r*v1[1] +  0.00*r*v2[1], p[2] + -1.00*r*v1[2] +  0.00*r*v2[2]);
      glVertex3d(p[0] + -0.71*r*v1[0] + -0.71*r*v2[0], p[1] + -0.71*r*v1[1] + -0.71*r*v2[1], p[2] + -0.71*r*v1[2] + -0.71*r*v2[2]);
      glVertex3d(p[0] +  0.00*r*v1[0] + -1.00*r*v2[0], p[1] +  0.00*r*v1[1] + -1.00*r*v2[1], p[2] +  0.00*r*v1[2] + -1.00*r*v2[2]);
      glVertex3d(p[0] +  0.71*r*v1[0] + -0.71*r*v2[0], p[1] +  0.71*r*v1[1] + -0.71*r*v2[1], p[2] +  0.71*r*v1[2] + -0.71*r*v2[2]);
      glEnd();
    }
    else {
      fprintf(stderr, "Unrecognized light type: %d\n", light->type);
      return;
    }
  }

  // Clean up
  if (lighting) glEnable(GL_LIGHTING);
}



void DrawCamera(R3Scene *scene)
{
  // Check if should draw lights
  if (!show_camera) return;

  // Setup
  GLboolean lighting = glIsEnabled(GL_LIGHTING);
  glDisable(GL_LIGHTING);
  glColor3d(1.0, 1.0, 1.0);
  glLineWidth(5);

  // Draw view frustum
  R3Camera& c = scene->camera;
  double radius = scene->bbox.DiagonalRadius();
  R3Point org = c.eye + c.towards * radius;
  R3Vector dx = c.right * radius * tan(c.xfov);
  R3Vector dy = c.up * radius * tan(c.yfov);
  R3Point ur = org + dx + dy;
  R3Point lr = org + dx - dy;
  R3Point ul = org - dx + dy;
  R3Point ll = org - dx - dy;
  glBegin(GL_LINE_LOOP);
  glVertex3d(ur[0], ur[1], ur[2]);
  glVertex3d(ul[0], ul[1], ul[2]);
  glVertex3d(ll[0], ll[1], ll[2]);
  glVertex3d(lr[0], lr[1], lr[2]);
  glVertex3d(ur[0], ur[1], ur[2]);
  glVertex3d(c.eye[0], c.eye[1], c.eye[2]);
  glVertex3d(lr[0], lr[1], lr[2]);
  glVertex3d(ll[0], ll[1], ll[2]);
  glVertex3d(c.eye[0], c.eye[1], c.eye[2]);
  glVertex3d(ul[0], ul[1], ul[2]);
  glEnd();

  // Clean up
  glLineWidth(1);
  if (lighting) glEnable(GL_LIGHTING);
}



void DrawScene(R3Scene *scene) 
{
  // Draw nodes recursively
  DrawNode(scene, scene->root);
}



////////////////////////////////////////////////////////////
// GLUT USER INTERFACE CODE
////////////////////////////////////////////////////////////

void GLUTMainLoop(void)
{
  // Run main loop -- never returns 
  glutMainLoop();
}


void GLUTStop(void)
{
  // Destroy window 
  glutDestroyWindow(GLUTwindow);

  // Delete scene
  delete scene;

  // Exit
  exit(0);
}



void GLUTResize(int w, int h)
{
  // Resize window
  glViewport(0, 0, w, h);

  // Resize camera vertical field of view to match aspect ratio of viewport
  camera.yfov = atan(tan(camera.xfov) * (double) h/ ((double) w/2)); 
  camera2.yfov = atan(tan(camera2.xfov) * (double) h/ ((double) w/2)); 

  // Remember window size 
  GLUTwindow_width = w;
  GLUTwindow_height = h;

  // Redraw
  glutPostRedisplay();
}
 

//INTERSECTION CODE
bool inside_box (R3Node* node, R3Point point) {
	bool intersect = false;
	if (node->shape != NULL) {
		if (node->shape->type == R3_BOX_SHAPE) {
			bool x = false;
			bool y = false;
			bool z = false;
			R3Box* box = node->shape->box;
			if ((point.X() >= box->XMin()) && (point.X() <= box->XMax())) {
				x = true;
			}
			if ((point.Y() >= box->YMin()) && (point.Y() <= box->YMax())) {
				y = true;
			}
			if ((point.Z() >= box->ZMin()) && (point.Z() <= box->ZMax())) {
				z = true;
			}
			if (x && y && z) {
				intersect = true;
				//puts("INTERSECT");
			}
		}
	}
	for (int i = 0; i < (int)node->children.size(); i++) {
		R3Node* child = node->children[i];
		if (inside_box(child, point)) intersect = true;
	}
	return intersect;
}

bool inside_tank (R3Box box, R3Point point) {
	bool intersect = false;

		bool x = false;
		bool y = false;
		bool z = false;
		if ((point.X() >= box.XMin()) && (point.X() <= box.XMax())) {
			x = true;
		}
		if ((point.Y() >= box.YMin()) && (point.Y() <= box.YMax())) {
			y = true;
		}
		if ((point.Z() >= box.ZMin()) && (point.Z() <= box.ZMax())) {
			z = true;
		}
		if (x && y && z) {
			intersect = true;
		}
	return intersect;
}

int intersect_bbox (R3Scene* scene, Tank* tank) {
	for (int i = 0; i < (int)tank->corners.size(); i++) {
		if (inside_box(scene->root, tank->corners[i])) {
			return i+1;
		}
		if (inside_tank(scene->redTank.shape->mesh->bbox, tank->corners[i])) {
			return i+1;
		}

	}
	return 0;
}

int intersect_bbox_red (R3Scene* scene, Tank* tank) {
  
	for (int i = 0; i < (int)tank->corners.size(); i++) {
		if (inside_box(scene->root, tank->corners[i])) {
			return i+1;
		}

		if (inside_tank(scene->blueTank.shape->mesh->bbox, tank->corners[i])) {
			return i+1;
		}

	}
	return 0;
}

bool intersect_base(R3Scene* scene, Tank* tank) {
	R3Point point (-13.5, 0, 13.5);
	bool intersect = false;
	R3Box* box = &tank->shape->mesh->bbox; 
	
	bool x = false;
	bool z = false;
	if ((point.X() >= box->XMin()) && (point.X() <= box->XMax())) {
		x = true;
	}
	if ((point.Z() >= box->ZMin()) && (point.Z() <= box->ZMax())) {
		z = true;
	}
	if (x && z) {
		intersect = true;
	}
	return intersect;
}

bool intersect_base_red(R3Scene* scene, Tank* tank) {
	R3Point point (-13.5, 0, -13.5);
	bool intersect = false;
	R3Box* box = &tank->shape->mesh->bbox; 
	
	bool x = false;
	bool z = false;
	if ((point.X() >= box->XMin()) && (point.X() <= box->XMax())) {
		x = true;
	}
	if ((point.Z() >= box->ZMin()) && (point.Z() <= box->ZMax())) {
		z = true;
	}
	if (x && z) {
		intersect = true;
	}
	return intersect;
}

bool intersect_flag(R3Scene* scene, Tank* tank) {
	R3Point point = scene->greenFlag.shape->mesh->bbox.Centroid();
	bool intersect = false;
	R3Box* box = &tank->shape->mesh->bbox; 
	
	bool x = false;
	bool z = false;
	if ((point.X() >= box->XMin()) && (point.X() <= box->XMax())) {
		x = true;
	}
	if ((point.Z() >= box->ZMin()) && (point.Z() <= box->ZMax())) {
		z = true;
	}
	if (x && z) {
		intersect = true;
	}
	return intersect;
}

// returns the t for a ray-plane intersection
double IntersectPlane(R3Ray ray, R3Plane p) {
	R3Point P0 = ray.Start();
	double numerator = -(P0.Vector().Dot(p.Normal()) + p.D());
	double denominator = ray.Vector().Dot(p.Normal());
	if (denominator == 0)
		return -1;
	return numerator/denominator;
}

R3Intersection ray_intersect_box(R3Ray ray, R3Box box) {
	R3Intersection i;
	i.hit = false;
	i.t = DBL_MAX;
	//R3Box *box = node->shape->box;
	// check each side of the box
	R3Point bbL = box.Min(); // stands for front bottom left
	R3Point bbR = box.Min();
	bbR.SetX(box.Max().X());
	R3Point btL = box.Min();
	btL.SetY(box.Max().Y());
	R3Point btR = box.Max();
	btR.SetZ(box.Min().Z());
	R3Point fbL = box.Min();
	fbL.SetZ(box.Max().Z());
	R3Point fbR = box.Max();
	fbR.SetY(box.Min().Y());
	R3Point ftL = box.Max();
	ftL.SetX(box.Min().X());
	R3Point ftR = box.Max();
	R3Vector frontV(0, 0, 1);
	R3Vector backV(0, 0, -1);
	R3Vector leftV(-1, 0, 0);
	R3Vector rightV(1, 0, 0);
	R3Vector topV(0, 1, 0);
	R3Vector bottomV(0, -1, 0);
	
	R3Plane front(fbL, frontV);
	if (ray.Vector().Dot(front.Normal()) < 0) {
		double t = IntersectPlane(ray, front);
		R3Point p = ray.Point(t);
		if (p.X() > fbL.X() && p.X() < fbR.X() && t > 0) {
			if (p.Y() > fbL.Y() && p.Y() < ftL.Y()) {
				if (i.hit && t < i.t || !i.hit) {
					i.hit = true;
					i.t = t;
					i.position = p;
					i.normal = front.Normal();
				}
			}
		}
	}
	
	R3Plane back(bbL, backV);
	if (ray.Vector().Dot(back.Normal()) < 0) {
		double t = IntersectPlane(ray, back);
		R3Point p = ray.Point(t);
		if (p.X() > bbL.X() && p.X() < bbR.X() && t > 0) {
			if (p.Y() > bbL.Y() && p.Y() < btL.Y()) {
				if (i.hit && t < i.t || !i.hit) {
					i.hit = true;
					i.t = t;
					i.position = p;
					i.normal = back.Normal();
				}
			}
		}
	}
	
	R3Plane left(bbL, leftV);
	if (ray.Vector().Dot(left.Normal()) < 0) {
		double t = IntersectPlane(ray, left);
		R3Point p = ray.Point(t);
		if (p.Z() > bbL.Z() && p.Z() < fbL.Z() && t > 0) {
			if (p.Y() > bbL.Y() && p.Y() < btL.Y()) {
				if (i.hit && t < i.t || !i.hit) {
					i.hit = true;
					i.t = t;
					i.position = p;
					i.normal = left.Normal();
				}
			}
		}
	}
	R3Plane right(bbR, rightV);
	if (ray.Vector().Dot(right.Normal()) < 0) {
		double t = IntersectPlane(ray, right);
		R3Point p = ray.Point(t);
		if (p.Z() > bbR.Z() && p.Z() < fbR.Z() && t > 0) {
			if (p.Y() > bbR.Y() && p.Y() < btR.Y()) {
				if (i.hit && t < i.t || !i.hit) {
					i.hit = true;
					i.t = t;
					i.position = p;
					i.normal = right.Normal();
				}
			}
		}
	}
	R3Plane bottom(bbL, bottomV);
	if (ray.Vector().Dot(bottom.Normal()) <= 0) {
		double t = IntersectPlane(ray, bottom);
		R3Point p = ray.Point(t);
		if (p.X() > bbL.X() && p.X() < fbR.X() && t > 0) {
			if (p.Z() > bbL.Z() && p.Z() < fbL.Z()) {
				if (i.hit && t < i.t || !i.hit) {
					i.hit = true;
					i.t = t;
					i.position = p;
					i.normal = bottom.Normal();
				}
			}
		}
	}
	R3Plane top(btL, topV);
	if (ray.Vector().Dot(top.Normal()) <= 0) {
		double t = IntersectPlane(ray, top);
		R3Point p = ray.Point(t);
		if (p.X() > btL.X() && p.X() < ftR.X() && t > 0) {
			if (p.Z() > btL.Z() && p.Z() < ftL.Z()) {
				if (i.hit && t < i.t || !i.hit) {
					i.hit = true;
					i.t = t;
					i.position = p;
					i.normal = top.Normal();
				}
			}
		}
	}
	/*if (i->hit) {
		i->node = node;
	}*/
	return i;
		
}

void GLUTMotion(int x, int y)
{
  // Invert y coordinate
  y = GLUTwindow_height - y;
  
  // Compute mouse movement
  int dx = x - GLUTmouse[0];
  int dy = y - GLUTmouse[1];
  // Process mouse motion event
  if ((dx != 0) || (dy != 0)) {
    R3Point scene_center = scene->bbox.Centroid();
    if ((GLUTbutton[0] && (GLUTmodifiers & GLUT_ACTIVE_SHIFT)) || GLUTbutton[1]) {
      // Scale world 
      double factor = (double) dx / (double) GLUTwindow_width;
      factor += (double) dy / (double) GLUTwindow_height;
      factor = exp(2.0 * factor);
      factor = (factor - 1.0) / factor;
      R3Vector translation = (scene_center - camera.eye) * factor;
      camera.eye += translation;
      glutPostRedisplay();
    }
    else if (GLUTbutton[0] && (GLUTmodifiers & GLUT_ACTIVE_CTRL)) {
      // Translate world
      double length = R3Distance(scene_center, camera.eye) * tan(camera.yfov);
      double vx = length * (double) dx / (double) GLUTwindow_width;
      double vy = length * (double) dy / (double) GLUTwindow_height;
      R3Vector translation = -((camera.right * vx) + (camera.up * vy));
      camera.eye += translation;
      glutPostRedisplay();
    }
    else if (GLUTbutton[0]) {
      // Rotate world
      double vx = (double) dx / (double) GLUTwindow_width;
      double vy = (double) dy / (double) GLUTwindow_height;
      double theta = 4.0 * (fabs(vx) + fabs(vy));
      R3Vector vector = (camera.right * vx) + (camera.up * vy);
      R3Vector rotation_axis = camera.towards % vector;
      rotation_axis.Normalize();
      camera.eye.Rotate(R3Line(scene_center, rotation_axis), theta);
      camera.towards.Rotate(rotation_axis, theta);
      camera.up.Rotate(rotation_axis, theta);
      camera.right = camera.towards % camera.up;
      camera.up = camera.right % camera.towards;
      camera.towards.Normalize();
      camera.up.Normalize();
      camera.right.Normalize();
      glutPostRedisplay();
    }
  }

  // Remember mouse position 
  GLUTmouse[0] = x;
  GLUTmouse[1] = y;
}

void apply_turn(double amount) {
	R3Vector up(0, 1, 0);
	R3Point centroid = scene->blueTank.shape->mesh->bbox.Centroid();
	R3Line lUp(centroid, up);
	scene->blueTank.shape->mesh->Rotate(amount, lUp);
	if (scene->blueTank.hasFlag) {
		scene->greenFlag.shape->mesh->Rotate(amount, lUp);
	}
	scene->blueTank.towards.Rotate(up, amount);
	scene->blueTank.right.Rotate(up, amount);
	for (int i = 0; i < (int)scene->blueTank.corners.size(); i++) {
		scene->blueTank.corners[i].Rotate(lUp, amount);
	}
	int intersection = intersect_bbox(scene, &scene->blueTank);
	if (intersection != 0) {
		amount = -amount;
		scene->blueTank.shape->mesh->Rotate(amount, lUp);
		if (amount < 0 && intersection == 2) {
		  R3Vector translation = scene->blueTank.right * -0.1;
		  scene->blueTank.shape->mesh->Translate(translation.X(), translation.Y(), translation.Z());
		  for (int i = 0; i < (int)scene->blueTank.corners.size(); i++) {
		    scene->blueTank.corners[i] += translation;
		  }
		  if (scene->blueTank.hasFlag) {
		    scene->greenFlag.shape->mesh->Translate(translation.X(), translation.Y(), translation.Z());
		  }
		}
		if (amount > 0 && intersection == 1) {
		  R3Vector translation = scene->blueTank.right * 0.1;
		  scene->blueTank.shape->mesh->Translate(translation.X(), translation.Y(), translation.Z());
		  for (int i = 0; i < (int)scene->blueTank.corners.size(); i++) {
		    scene->blueTank.corners[i] += translation;
		  }
		  if (scene->blueTank.hasFlag) {
		    scene->greenFlag.shape->mesh->Translate(translation.X(), translation.Y(), translation.Z());
		  }
		}
		if (scene->blueTank.hasFlag) {
			scene->greenFlag.shape->mesh->Rotate(amount, lUp);
		}
		scene->blueTank.towards.Rotate(up, amount);
		scene->blueTank.right.Rotate(up, amount);
		for (int i = 0; i < (int)scene->blueTank.corners.size(); i++) {
			scene->blueTank.corners[i].Rotate(lUp, amount);
		}
	}
	else {
			scene->blueTank.rotated += amount;
	}
	glutPostRedisplay();
}

void apply_turn_red(double amount) {
	R3Vector up(0, 1, 0);
	R3Point centroid = scene->redTank.shape->mesh->bbox.Centroid();
	R3Line lUp(centroid, up);
	scene->redTank.shape->mesh->Rotate(amount, lUp);
	if (scene->redTank.hasFlag) {
		scene->greenFlag.shape->mesh->Rotate(amount, lUp);
	}
	scene->redTank.towards.Rotate(up, amount);
	scene->redTank.right.Rotate(up, amount);
	for (int i = 0; i < (int)scene->redTank.corners.size(); i++) {
		scene->redTank.corners[i].Rotate(lUp, amount);
	}
		int intersection = intersect_bbox_red(scene, &scene->redTank);
	if (intersection != 0) {
		amount = -amount;
		scene->redTank.shape->mesh->Rotate(amount, lUp);
		if (amount < 0 && intersection == 2) {
		  R3Vector translation = scene->redTank.right * -0.1;
		  scene->redTank.shape->mesh->Translate(translation.X(), translation.Y(), translation.Z());
		  for (int i = 0; i < (int)scene->redTank.corners.size(); i++) {
		    scene->redTank.corners[i] += translation;
		  }
		  if (scene->redTank.hasFlag) {
		  scene->greenFlag.shape->mesh->Translate(translation.X(), translation.Y(), translation.Z());
		  }
		}
		if (amount > 0 && intersection == 1) {
		  R3Vector translation = scene->redTank.right * 0.1;
		  scene->redTank.shape->mesh->Translate(translation.X(), translation.Y(), translation.Z());
		  for (int i = 0; i < (int)scene->redTank.corners.size(); i++) {
		    scene->redTank.corners[i] += translation;
		  }
		  if (scene->redTank.hasFlag) {
		    scene->greenFlag.shape->mesh->Translate(translation.X(), translation.Y(), translation.Z());
		  }
		}
		if (scene->redTank.hasFlag) {
			scene->greenFlag.shape->mesh->Rotate(amount, lUp);
		}
		scene->redTank.towards.Rotate(up, amount);
		scene->redTank.right.Rotate(up, amount);
		for (int i = 0; i < (int)scene->redTank.corners.size(); i++) {
			scene->redTank.corners[i].Rotate(lUp, amount);
		}
	}
	else {
    scene->redTank.rotated += amount;
	}
	glutPostRedisplay();
}

 void create_bullet() {
	 if (scene->blueTank.bulletDelay > 1) {
		 Bullet *b = new Bullet();
		 b->towards = scene->blueTank.towards;
		 b->lifetime = 0.0;
		 b->material = scene->bulletMaterial;
		 b->color = 0;
		 R3Shape *shape = new R3Shape();
		 shape->type = R3_SPHERE_SHAPE;
		 shape->box = NULL;
		 shape->sphere = new R3Sphere(scene->blueTank.shape->mesh->bbox.Centroid() + b->towards*.5, .05);
		 shape->cylinder = NULL;
		 shape->cone = NULL;
		 shape->mesh = NULL;
		 shape->segment = NULL;
		 b->shape = shape;
		 scene->bullets.push_back(b);
		 scene->blueTank.bulletDelay = 0.0;
	 }
 }
 void create_bullet_red() {
	 //fprintf(stderr, "%f\n", scene->redTank.bulletDelay);
	 if (scene->redTank.bulletDelay > 1) {
		 //fprintf(stderr, "creating bullet\n");
		 Bullet *b = new Bullet();
		 b->towards = scene->redTank.towards;
		 b->lifetime = 0.0;
		 b->material = scene->bulletMaterial;
		 b->color = 1;
		 R3Shape *shape = new R3Shape();
		 shape->type = R3_SPHERE_SHAPE;
		 shape->box = NULL;
		 shape->sphere = new R3Sphere(scene->redTank.shape->mesh->bbox.Centroid() + b->towards*.5, .05);
		 shape->cylinder = NULL;
		 shape->cone = NULL;
		 shape->mesh = NULL;
		 shape->segment = NULL;
		 //fprintf(stderr, "%f %f %f\n", shape->sphere->Center().X(), shape->sphere->Center().Y(), shape->sphere->Center().Z());
		 b->shape = shape;
		 scene->bullets.push_back(b);
		 scene->redTank.bulletDelay = 0.0;
	 }
 }

 void delete_bullet(int index) {
	 Bullet *b = scene->bullets[index];
	 scene->bullets[index] = scene->bullets.back();
	 scene->bullets.pop_back();
	 delete b->shape->sphere;
	 delete b->shape;
	 delete b;
 }
 
 void draw_bullets() {
	
	 for (int i = (int) scene->bullets.size() -1; i >= 0; i--) {
		 if (scene->bullets[i]->color == 0)
			 LoadMaterial(scene->blueTank.material);
		 else
			 LoadMaterial(scene->redTank.material);
		 //fprintf(stderr, "%f\n", scene->bullets[i]->shape->sphere->Center().Z());
		 DrawShape(scene->bullets[i]->shape);
		 glutPostRedisplay();
	 }
	// glutPostRedisplay();
 }

 void reset_flag() {
  R3Mesh *m = scene->greenFlag.shape->mesh;
	srand(time(NULL));
	int number = rand()%3 + 1;
	if (number == 1) {
		 m->Translate(14 - m->bbox.Centroid().X(), 0, -10 - m->bbox.Centroid().Z());
	}
	else if (number == 2) {
		 m->Translate(14 - m->bbox.Centroid().X(), 0, 0 - m->bbox.Centroid().Z());
	}
	else {
		 m->Translate(14 - m->bbox.Centroid().X(), 0, 10 - m->bbox.Centroid().Z());
	}
 
  scene->greenFlag.shape->mesh = m;
 }

 void reset_tank() {
	engine->play2D("../sound/explosion.wav", false);
  R3Mesh *m = scene->blueTank.shape->mesh;

  for (int i = 0; i < (int)scene->blueTank.corners.size();i++) {
 	 scene->blueTank.corners[i] = R3Point(scene->blueTank.corners[i].X() -13.5 - m->bbox.Centroid().X(), scene->blueTank.corners[i].Y(), scene->blueTank.corners[i].Z() +13.5 - m->bbox.Centroid().Z());
  }
  m->Translate(-13.5 - m->bbox.Centroid().X(), 0, 13.5 - m->bbox.Centroid().Z());
  scene->blueTank.shape->mesh = m;
  
  apply_turn(scene->blueTank.rotated * -1);
  // reset camera
  camera.eye = R3Point(-22, 8.5, 13.5);
 	camera.towards = scene->blueTank.shape->mesh->bbox.Centroid() - camera.eye + R3Vector(0, 1.8, 0);
 	camera.towards.Normalize();
 	camera.right.Reset(0, 0, 1);
 	camera.xz.Reset(1, 0, 0);
 	camera.up = camera.right;
 	camera.up.Cross(camera.towards);
 	camera.up.Normalize();
 	scene->blueTank.health = 100;
 }
 void reset_tank_red() {
	engine->play2D("../sound/explosion.wav", false);
  R3Mesh *m = scene->redTank.shape->mesh;

  for (int i = 0; i < (int)scene->redTank.corners.size();i++) {
 	 scene->redTank.corners[i] = R3Point(scene->redTank.corners[i].X() -13.5 - m->bbox.Centroid().X(), scene->redTank.corners[i].Y(), scene->redTank.corners[i].Z() - 13.5 - m->bbox.Centroid().Z());
  }
  m->Translate(-13.5 - m->bbox.Centroid().X(), 0, -13.5 - m->bbox.Centroid().Z());
  scene->redTank.shape->mesh = m;
  
  apply_turn_red(scene->redTank.rotated * -1);
  // reset camera
  camera2.eye = R3Point(-22, 8.5, -13.5);
 	camera2.towards = scene->redTank.shape->mesh->bbox.Centroid() - camera2.eye + R3Vector(0, 1.8, 0);
 	camera2.towards.Normalize();
 	camera2.right.Reset(0, 0, 1);
 	camera2.xz.Reset(1, 0, 0);
 	camera2.up = camera2.right;
 	camera2.up.Cross(camera2.towards);
 	camera2.up.Normalize();
 	scene->redTank.health = 100;
 }
 
 void transparent_walls() {
	 R3Intersection intersect;
	 R3Node *intersectedNode;
	 intersect.hit = false;
	 intersect.t = DBL_MAX;
	 R3Ray ray(camera.eye, scene->blueTank.shape->mesh->bbox.Centroid() - camera.eye);
	 R3Box tankBox(scene->blueTank.shape->mesh->bbox);
	 R3Intersection tankHit = ray_intersect_box(ray, tankBox);
	 if (tankHit.hit) intersect = tankHit;
	 for (int j = 0; j < (int)scene->root->children.size(); j++) {
		 if (scene->root->children[j]->shape->type == R3_BOX_SHAPE) {
			 R3Intersection hit = ray_intersect_box(ray, *(scene->root->children[j]->shape->box));
			 // found hit
			 if (hit.hit && hit.t < intersect.t) {
				 intersect = hit;
				 intersectedNode = scene->root->children[j];
			 }
		 }
	 }
	 if (intersect.hit && intersect.t != tankHit.t) {
		 R3Material *trans = new R3Material(*intersectedNode->material);
		 trans->kt = R3Rgb(.6, .6, .6, .6);
		 intersectedNode->material = trans;
		 scene->transparentWalls.push_back(intersectedNode);
	 }
 }
 void transparent_walls2() {
	 R3Intersection intersect;
	 R3Node *intersectedNode;
	 intersect.hit = false;
	 intersect.t = DBL_MAX;
	 // ray pointing at the centroid
	 R3Ray ray(camera2.eye, scene->redTank.shape->mesh->bbox.Centroid() - camera2.eye);
	 R3Box tankBox(scene->redTank.shape->mesh->bbox);
	 R3Intersection tankHit = ray_intersect_box(ray, tankBox);
	 if (tankHit.hit) {
		 intersect = tankHit;
	 }
	 for (int j = 0; j < (int)scene->root->children.size(); j++) {
		 if (scene->root->children[j]->shape->type == R3_BOX_SHAPE) {
			 R3Intersection hit = ray_intersect_box(ray, *(scene->root->children[j]->shape->box));
			 // found hit
			 if (hit.hit && hit.t < intersect.t) {
				 intersect = hit;
				 intersectedNode = scene->root->children[j];
			 }
		 }
	 }
	 if (intersect.hit && intersect.t != tankHit.t) {
		 R3Material *trans = new R3Material(*intersectedNode->material);
		 trans->kt = R3Rgb(.6, .6, .6, .6);
		 intersectedNode->material = trans;
		 scene->transparentWalls.push_back(intersectedNode);
		 //fprintf(stderr, "%d\n", (int)transparentWalls.size());
	 }
 }
 void normal_walls() {
	 for (int i = scene->transparentWalls.size() -1; i >= 0; i--) {
		 scene->transparentWalls[i]->material->kt = R3Rgb(0, 0, 0, 0);
		 scene->transparentWalls.pop_back();
	 }
 }
 

 void create_particles(int tank, int amount) {
	 for (int i = 0; i < amount; i++) {
		 R3Particle *particle = new R3Particle();
		 particle->velocity = R3Vector(0, 1, 0);
		 // set up random position in the circle of the tank
		 R3Vector up(0, 1, 0);
		R3Vector tangent(1, 0, 0);
		double angle = 2*M_PI*rand() /(double)RAND_MAX;
		tangent.Rotate(up, angle);
		double randomFact = rand() /(double)RAND_MAX * .2;
		R3Point center;
		if (tank == 0)
			center = scene->blueTank.shape->mesh->bbox.Centroid();
		else if (tank == 1)
			center = scene->redTank.shape->mesh->bbox.Centroid();
		R3Point randomP = tangent.Point()*randomFact;
		randomP += center;
		particle->position = randomP;
		
		// rotate a random angle
		R3Vector A(0, 0, -.1);
		
		double t1 = rand() /(double)RAND_MAX* 2 * M_PI;
		double t2 = rand() /(double)RAND_MAX * sin(.3);
		A.Rotate(particle->velocity, t1);
		R3Vector axis = A;
		axis.Cross(particle->velocity);
		A.Rotate(axis, acos(1-t2));
		A += R3Vector(0, .6, 0);
		A.Normalize();
		double randFactor = (rand()/ (double)RAND_MAX + 1)/ 2;
		particle->velocity = 7*A * randFactor;
		particle->mass = .1;
		particle->drag = .1;
		particle->color = tank;
		particle->elasticity = .5;
		particle->material = scene->blueTank.material;
		particle->lifetime = 0.0;
		 R3Shape *shape = new R3Shape();
		 shape->type = R3_SPHERE_SHAPE;
		 shape->box = NULL;
		 shape->sphere = new R3Sphere(particle->position, .02);
		 shape->cylinder = NULL;
		 shape->cone = NULL;
		 shape->mesh = NULL;
		 shape->segment = NULL;
		 particle->shape = shape;
		scene->particles.push_back(particle);
	 }
 }
 void update_particles() {
	 for (int i = (int)scene->particles.size() - 1; i >= 0; i--) {
		 bool deleted = false;
		R3Particle *p = scene->particles[i];
		R3Point pos = p->position;
		R3Vector v = p->velocity;
		double mass = p->mass;
		R3Vector force(0, 0, 0);
		R3Vector gravity(0, -8, 0);
		R3Vector blank(0, 0, 0);
		// account for drag
		double k = -1.0*p->drag;
		if (v != blank)
			force += v * k;
		force += gravity * mass;

		// check lifetime
		if (p->lifetime >1) {
			deleted = true;
		}
		if (deleted) {
			R3Particle *p = scene->particles[i];
			scene->particles[i] = scene->particles.back();
			//vertices[i]->id = i;
			scene->particles.pop_back();
		
			// Delete particle
			delete p;
		}
		else {
			p->lifetime += .045;
			R3Point newPos = pos + .06*v.Point();
			
			R3Ray ray(pos, v);
			double newT = ray.T(newPos);
			R3Intersection intersect;
			intersect.hit = false;
			intersect.t = DBL_MAX;
			double collisionT = DBL_MAX - 1;
			
			 for (int j = 0; j < (int)scene->root->children.size(); j++) {
				 if (scene->root->children[j]->shape->type == R3_BOX_SHAPE) {
				   R3Intersection hit = ray_intersect_box(ray, *(scene->root->children[j]->shape->box));
					 // found hit
					 if (hit.hit && hit.t < intersect.t) {
						 intersect = hit;
						 collisionT = ray.T(intersect.position); 
					 }
				 }
			 }
				
			while (collisionT < newT && collisionT > 0 && intersect.hit) {
				newPos = intersect.position + intersect.normal*.01;
				R3Vector N = intersect.normal;
				// calculate the mirror of Lto across N
				R3Vector R = v - 2*v.Dot(N) * N;
				R3Vector parallelR = R;
				parallelR.Project(N);
				R3Vector orthR = R - parallelR;
				parallelR *= .5;
				v = parallelR + orthR;
				ray = R3Ray(newPos, v);
				intersect.hit = false;
				intersect.t = DBL_MAX;
				for (int j = 0; j < (int)scene->root->children.size(); j++) {
					 if (scene->root->children[j]->shape->type == R3_BOX_SHAPE) {
					   R3Intersection hit = ray_intersect_box(ray, *(scene->root->children[j]->shape->box));
						 // found hit
						 if (hit.hit && hit.t < intersect.t) {
							 intersect = hit;
							 collisionT += ray.T(intersect.position); 
						 }
					 }
				 }
			}
			// check if collision is going to occur
			//R3Intersection *intersect = ComputeIntersection(scene, ray);
			//double collisionT = ray->T(intersect->position);
			/*
			while (collisionT < newT && collisionT > 0 && intersect->hit) {
				newPos = intersect->position + intersect->normal*.01;
				R3Vector N = intersect->normal;
				// calculate the mirror of Lto across N
				R3Vector R = v - 2*v.Dot(N) * N;
				R3Vector parallelR = R;
				parallelR.Project(N);
				R3Vector orthR = R - parallelR;
				parallelR *= p->elasticity;
				v = parallelR + orthR;
				ray = new R3Ray(newPos, v);
				intersect = ComputeIntersection(scene, ray);
				collisionT += ray->T(intersect->position);
			}*/
			

			
			R3Vector newV = v + .06*force/mass;
			p->position = newPos;
			p->shape->sphere->Reposition(p->position);
			p->velocity = newV;
			if (p->color == 0)
				LoadMaterial(scene->blueTank.material);
			else
				LoadMaterial(scene->redTank.material);
			scene->particles[i] = p;
			p->shape->sphere->BBox().Draw();
		}	
	}
	 glutPostRedisplay();
 }
	 
	
 void update_bullets() {
	 scene->blueTank.bulletDelay += .065;
	 scene->redTank.bulletDelay += .065;
	 for (int i = (int)scene->bullets.size() -1; i >= 0; i--) {
		 R3Point newPosition = scene->bullets[i]->shape->sphere->Center() + scene->bullets[i]->towards *.17;
		 // check for collisions
		 R3Intersection intersect;
		 intersect.hit = false;
		 intersect.t = DBL_MAX;
		
		 R3Ray ray(scene->bullets[i]->shape->sphere->Center(), scene->bullets[i]->towards);
		 // first check tank hit
		 R3Box tankBox(scene->blueTank.shape->mesh->bbox);
		 R3Box redBox(scene->redTank.shape->mesh->bbox);
		 R3Intersection tankHit = ray_intersect_box(ray, tankBox);
		 R3Intersection redHit = ray_intersect_box(ray, redBox);
		 if (tankHit.hit && ray.T(tankHit.position) < ray.T(newPosition)) {
		   scene->blueTank.health -= 18 * ((1 - scene->bullets[i]->lifetime) * .6 + .4);
			engine->play2D("../sound/hit.wav", false);
		   if (scene->blueTank.health <= 0) {
				if (scene->blueTank.hasFlag) {
					scene->greenFlag.shape->mesh->Translate(0, -.3, 0);
					scene->blueTank.hasFlag = false;
					reset_flag();
				}
				create_particles(0, 300);
				reset_tank();

		   }
		   else {
			   double r = rand()/(double)RAND_MAX;
			   int amount = 20 * r;
			   create_particles(0, amount);
		   }
		   delete_bullet(i);
		   break;
		 }
		 else if (redHit.hit && ray.T(redHit.position) < ray.T(newPosition)) {
		   scene->redTank.health -= 18 * ((1 - scene->bullets[i]->lifetime) * .6 + .4);
			engine->play2D("../sound/hit.wav", false);
		   if (scene->redTank.health <= 0) {
				if (scene->redTank.hasFlag) {
					scene->greenFlag.shape->mesh->Translate(0, -.3, 0);
					scene->redTank.hasFlag = false;
					reset_flag();
				}
				create_particles(1, 300);
				reset_tank_red();

		   }
		   else {
			   double r = rand()/(double)RAND_MAX;
			   int amount = 20 * r;
			   create_particles(1, amount);
		   }
		   delete_bullet(i);
		   break;
		 }
		 else {
			 for (int j = 0; j < (int)scene->root->children.size(); j++) {
				 if (scene->root->children[j]->shape->type == R3_BOX_SHAPE) {
				   R3Intersection hit = ray_intersect_box(ray, *(scene->root->children[j]->shape->box));
					 // found hit
					 if (hit.hit && hit.t < intersect.t) {
						 intersect = hit;
					 }
				 }
			 }
	
			 // bounce
			 if (intersect.hit && ray.T(intersect.position) < ray.T(newPosition)) {
				 R3Vector bounce = scene->bullets[i]->towards - 2*scene->bullets[i]->towards.Dot(intersect.normal) * intersect.normal;
				 scene->bullets[i]->shape->sphere->Translate(bounce * .17);
				 scene->bullets[i]->towards = bounce;
				 engine->play2D("../sound/hit.wav", false);
			 }
			 // or go forward
			 else
				 scene->bullets[i]->shape->sphere->Translate(scene->bullets[i]->towards * .17);
			 scene->bullets[i]->lifetime += .01;
			 
			 if (scene->bullets[i]->lifetime >= 1) {
				 delete_bullet(i);
			 }
		 }
	 }
	 draw_bullets();
 }
	/*
	camera.right.Rotate(up, -.028);
	camera.up.Rotate(up, -.028); 
	R3Vector eyeV = camera.eye - centroid + R3Vector(0, .2, 0);
	R3Vector rotateEyeV = eyeV;
	rotateEyeV.Rotate(up, -.028);
	camera.eye += rotateEyeV - eyeV;
	
	camera.towards = centroid - camera.eye;
	camera.towards.Normalize();
	*/

void apply_move_forwards() {
 	R3Vector move = .07*scene->blueTank.towards;
 	scene->blueTank.shape->mesh->Translate(move.X(), move.Y(), move.Z());
	if (scene->blueTank.hasFlag) {
		scene->greenFlag.shape->mesh->Translate(move.X(), move.Y(), move.Z());
	}
 	for (int i = 0; i < (int)scene->blueTank.corners.size(); i++) {
     scene->blueTank.corners[i] = scene->blueTank.corners[i] + move;
   }
   //camera.eye = camera.eye + .1*scene->blueTank.towards;
	if (intersect_bbox(scene, &scene->blueTank) != 0) {
		scene->blueTank.shape->mesh->Translate(-move.X(), -move.Y(), -move.Z());
		if (scene->blueTank.hasFlag) {
			scene->greenFlag.shape->mesh->Translate(-move.X(), -move.Y(), -move.Z());
		}
	 	for (int i = 0; i < (int)scene->blueTank.corners.size(); i++) {
	     scene->blueTank.corners[i] = scene->blueTank.corners[i] - move;
	   }
	}
   glutPostRedisplay();
}

void apply_move_forwards_red() {
 	R3Vector move = .07*scene->redTank.towards;
 	scene->redTank.shape->mesh->Translate(move.X(), move.Y(), move.Z());
	if (scene->redTank.hasFlag) {
		scene->greenFlag.shape->mesh->Translate(move.X(), move.Y(), move.Z());
	}
 	for (int i = 0; i < (int)scene->redTank.corners.size(); i++) {
     scene->redTank.corners[i] = scene->redTank.corners[i] + move;
   }
   //camera2.eye = camera2.eye + .1*scene->redTank.towards;
	if (intersect_bbox_red(scene, &scene->redTank) != 0) { 
		scene->redTank.shape->mesh->Translate(-move.X(), -move.Y(), -move.Z());
		if (scene->redTank.hasFlag) {
			scene->greenFlag.shape->mesh->Translate(-move.X(), -move.Y(), -move.Z());
		}
	 	for (int i = 0; i < (int)scene->redTank.corners.size(); i++) {
	     scene->redTank.corners[i] = scene->redTank.corners[i] - move;
	   }
		
	}
   glutPostRedisplay();
}

void apply_move_backwards() {
 	R3Vector move = -.08*scene->blueTank.towards;
 	scene->blueTank.shape->mesh->Translate(move.X(), move.Y(), move.Z());
	if (scene->blueTank.hasFlag) {
		scene->greenFlag.shape->mesh->Translate(move.X(), move.Y(), move.Z());
	}
 	for (int i = 0; i < (int)scene->blueTank.corners.size(); i++) {
     scene->blueTank.corners[i] = scene->blueTank.corners[i] + move;
   }
  	if (intersect_bbox(scene, &scene->blueTank) != 0) {
		scene->blueTank.shape->mesh->Translate(-move.X(), -move.Y(), -move.Z());
		if (scene->blueTank.hasFlag) {
			scene->greenFlag.shape->mesh->Translate(-move.X(), -move.Y(), -move.Z());
		}
	 	for (int i = 0; i < (int)scene->blueTank.corners.size(); i++) {
	     scene->blueTank.corners[i] = scene->blueTank.corners[i] - move;
	   }
	}
   glutPostRedisplay();
}
void apply_move_backwards_red() {
  	R3Vector move = -.08*scene->redTank.towards;
  	//fprintf(stderr, "moving backwards?\n");
  	scene->redTank.shape->mesh->Translate(move.X(), move.Y(), move.Z());
		if (scene->redTank.hasFlag) {
			scene->greenFlag.shape->mesh->Translate(move.X(), move.Y(), move.Z());
		}
  	for (int i = 0; i < (int)scene->redTank.corners.size(); i++) {
      scene->redTank.corners[i] = scene->redTank.corners[i] + move;
    }
    if (intersect_bbox_red(scene, &scene->redTank) != 0) { 
			scene->redTank.shape->mesh->Translate(-move.X(), -move.Y(), -move.Z());
			if (scene->redTank.hasFlag) {
				scene->greenFlag.shape->mesh->Translate(-move.X(), -move.Y(), -move.Z());
			}
		 	for (int i = 0; i < (int)scene->redTank.corners.size(); i++) {
		     scene->redTank.corners[i] = scene->redTank.corners[i] - move;
		   }

		}
    glutPostRedisplay();
}

void set_front_camera(int color) {
	Tank tank;
	if (color == 0) {
		tank = scene->blueTank;
		camera.towards = tank.towards;
		camera.eye = tank.shape->mesh->bbox.Centroid() - tank.towards * .16 + R3Vector(0, .3, 0);
		camera.right = tank.right;
		camera.up = R3Vector(0, 1, 0);
		camera.xz = camera.towards;
	}
	else {
		tank = scene->redTank;
		camera2.towards = tank.towards;
		camera2.eye = tank.shape->mesh->bbox.Centroid() - tank.towards * .16 + R3Vector(0, .3, 0);
		camera2.right = tank.right;
		camera2.up = R3Vector(0, 1, 0);
		camera2.xz = camera.towards;
	}

}


void rotate_camera(double amount) {
	R3Vector up(0, 1, 0);
	R3Point centroid = scene->blueTank.shape->mesh->bbox.Centroid();
	R3Vector eyeV = camera.eye - centroid + R3Vector(0, 1.8, 0);
	R3Vector rotateEyeV = eyeV;
	rotateEyeV.Rotate(up, amount);
	rotateEyeV.SetY(eyeV.Y());
	camera.eye += rotateEyeV - eyeV;
	
	
	camera.towards = centroid - camera.eye + R3Vector(0, 1.8, 0);
	camera.towards.Normalize();
	camera.xz = camera.towards;
	camera.xz.SetY(0);
	camera.xz.Normalize();
	camera.right = camera.towards;
	camera.right.Cross(camera.xz);
	camera.right.Normalize();
	camera.up = camera.right;
	camera.up.Cross(camera.towards);
}

void rotate_camera2(double amount) {
	R3Vector up(0, 1, 0);
	R3Point centroid = scene->redTank.shape->mesh->bbox.Centroid();
	R3Vector eyeV = camera2.eye - centroid + R3Vector(0, 1.8, 0);
	R3Vector rotateEyeV = eyeV;
	rotateEyeV.Rotate(up, amount);
	camera2.eye += rotateEyeV - eyeV;

	camera2.towards = centroid - camera2.eye + R3Vector(0, 1.8, 0);
	camera2.towards.Normalize();
	camera2.xz = camera2.towards;
	camera2.xz.SetY(0);
	camera2.xz.Normalize();
	camera2.right = camera2.towards;
	camera2.right.Cross(camera2.xz);
	camera2.right.Normalize();
	camera2.up = camera2.right;
	camera2.up.Cross(camera2.towards);
}

void update_camera_rotation() {
	// check if camera is inline with tank
	double angle = camera.xz.Dot(scene->blueTank.towards);
	//fprintf(stderr, "dot: %f, angle: %f\n", angle, acos(angle));
	// find the angle
	if (abs(1-angle) < .0001)
		angle = 0;
	else {
		angle = acos(angle);
		if (!angle) angle = 0;
	}

	if (!turning && scene->cameraUpdate == 1.0) {
		//fprintf(stderr, "RESET RESET RESET\n");
		scene->cameraUpdate = 0.0;
	}
	
	if (angle != 0) {
		// make the camera move more
		scene->cameraUpdate += .0025;
		// never let update get bigger than 1
		if (scene->cameraUpdate > 1) scene->cameraUpdate = 1;
		// move the camera closer to the tank
		angle *= scene->cameraUpdate;
		// find which side the camera vector is on
		R3Vector up = camera.xz;
		up.Cross(scene->blueTank.towards);
		//fprintf(stderr, "%f\n", scene->cameraUpdate);
		// rotate camera
		if (up.Y() > 0) {
			rotate_camera(angle);
		}
		else {
			rotate_camera(angle * -1);
		}
			
	}
	// if inline
	else {
		//fprintf(stderr, "in line\n");
		if (!turning)
			//fprintf(stderr, "RESET RESET RESET\n");
			scene->cameraUpdate = 0.0;
	}
	glutPostRedisplay();
}

void update_camera_position() {
	// check if camera is inline with tank
	R3Point cameraPoint = camera.eye;	

	R3Vector dist = scene->blueTank.shape->mesh->bbox.Centroid() - cameraPoint;
	R3Vector yDist = R3Vector(0, 8.5, 0) - R3Vector(0, camera.eye.Y(), 0);
	dist.SetY(0);
	double diff = dist.Length() - 8.5;
	R3Vector target = dist;
	target.Normalize();
	target *= 8.5;
	target = dist - target + yDist; 
	if (abs(diff) > .001) {
		// make the camera move more
		scene->cameraPosition += .0025;
		// never let update get bigger than 1
		if (scene->cameraPosition > 1) scene->cameraPosition = 1;
		// move the camera closer to the tank
		camera.eye += target * scene->cameraPosition;	
		// update towards and right and up vectors
		camera.towards = scene->blueTank.shape->mesh->bbox.Centroid() - camera.eye + R3Vector(0, 1.8, 0);
		camera.towards.Normalize();
		camera.xz = camera.towards;
		camera.xz.SetY(0);
		camera.xz.Normalize();
		camera.right = camera.towards;
		camera.right.Cross(camera.xz);
		camera.right.Normalize();
		camera.up = camera.right;
		camera.up.Cross(camera.towards);
	}
	// if inline
	else {
		scene->cameraPosition = 0.0;
	}
	glutPostRedisplay();
}

void update_camera2_position() {
	// check if camera is inline with tank
	R3Point cameraPoint = camera2.eye;
	R3Vector dist = scene->redTank.shape->mesh->bbox.Centroid() - cameraPoint;
	R3Vector yDist = R3Vector(0, 8.5, 0) - R3Vector(0, camera2.eye.Y(), 0);
  dist.SetY(0);
	double diff = dist.Length() - 8.5;
	R3Vector target = dist;
	target.Flip();
	target.Normalize();
	target *= 8.5;
	target = dist + target + yDist; 
	//fprintf(stderr, "%f\n", diff);
	if (abs(diff) > .001) {
		// make the camera move more
		scene->cameraPosition2 += .0025;
		// never let update get bigger than 1
		if (scene->cameraPosition2 > 1) scene->cameraPosition2 = 1;
		// move the camera closer to the tank
		diff *= scene->cameraPosition2;
		camera2.eye += target * scene->cameraPosition2;	
		camera2.towards = scene->redTank.shape->mesh->bbox.Centroid() - camera2.eye + R3Vector(0, 1.8, 0);
		camera2.towards.Normalize();
		camera2.xz = camera2.towards;
		camera2.xz.SetY(0);
		camera2.xz.Normalize();
		camera2.right = camera2.towards;
		camera2.right.Cross(camera2.xz);
		camera2.right.Normalize();
		camera2.up = camera2.right;
		camera2.up.Cross(camera2.towards);
	}
	// if inline
	else {
		scene->cameraPosition2 = 0.0;
	}
	glutPostRedisplay();
}

void update_camera2_rotation() {
	// check if camera is inline with tank
	double angle = camera2.xz.Dot(scene->redTank.towards);
	//fprintf(stderr, "dot: %f, angle: %f\n", angle, acos(angle));
	// find the angle
	if (abs(1-angle) < .0001)
		angle = 0;
	else {
		angle = acos(angle);
		if (!angle) angle = 0;
	}

	if (!turning && scene->cameraUpdate2 == 1.0) {
		//fprintf(stderr, "RESET RESET RESET\n");
		scene->cameraUpdate2 = 0.0;
	}
	
	if (angle != 0) {
		// make the camera move more
		scene->cameraUpdate2 += .0025;
		// never let update get bigger than 1
		if (scene->cameraUpdate2 > 1) scene->cameraUpdate2 = 1;
		// move the camera closer to the tank
		angle *= scene->cameraUpdate2;
		// find which side the camera vector is on
		R3Vector up = camera2.xz;
		up.Cross(scene->redTank.towards);
		//fprintf(stderr, "%f\n", scene->cameraUpdate);
		// rotate camera
		if (up.Y() > 0) {
			rotate_camera2(angle);
		}
		else {
			rotate_camera2(angle * -1);
		}
		if (scene->cameraUpdate2 == 1.0) {
			angle = camera2.xz.Dot(scene->redTank.towards);
			//fprintf(stderr, "SHOULD BE 1:   %f\n", angle);
		}
			
	}
	// if inline
	else {
		//fprintf(stderr, "in line\n");
		if (!turning)
			//fprintf(stderr, "RESET RESET RESET\n");
			scene->cameraUpdate2 = 0.0;
	}
	glutPostRedisplay();
}

//Enable and Disable Methods from 
//http://www.gamedev.net/topic/104791-last-post-about-2d-in-opengl-so-please-stop/
void glEnable2D()
{
	int vPort[4];

   glGetIntegerv(GL_VIEWPORT, vPort);

   glMatrixMode(GL_PROJECTION);
   glPushMatrix();
   glLoadIdentity();

   glOrtho(0, vPort[2], 0, vPort[3], -1, 1);
   glMatrixMode(GL_MODELVIEW);
   glPushMatrix();
   glLoadIdentity();
	 glDisable(GL_LIGHTING);
}

void glDisable2D()
{
   glMatrixMode(GL_PROJECTION);
   glPopMatrix();   
   glMatrixMode(GL_MODELVIEW);
   glPopMatrix();	
	glEnable(GL_LIGHTING);
}

void draw_hud(int tank_num) {
  Tank* this_tank;
  Tank* other_tank;
  glEnable2D();
  glBegin(GL_QUADS);
  glColor3f(0.0f,1.0f,0.0f);
  if (tank_num == 0) {
    this_tank = &(scene->blueTank);
    other_tank = &(scene->redTank);
  }
  else {
    this_tank = &(scene->redTank);
    other_tank = &(scene->blueTank);    
  }
  if (this_tank->health < 66) {
    glColor3f(1.0f,1.0f,0.0f);
  }
  if (this_tank->health < 33) {
    glColor3f(1.0f,0.0f,0.0f);
  }
  glVertex2d(10, GLUTwindow_height - 20);
  glVertex2d(10, GLUTwindow_height - 50);
  glVertex2d(10 + this_tank->health*2, GLUTwindow_height - 50);
  glVertex2d(10 + this_tank->health*2, GLUTwindow_height - 20);
  
  glColor3f(0.5f,0.5f,0.5f);
  
  glVertex2d(5, GLUTwindow_height - 15);
  glVertex2d(5, GLUTwindow_height - 55);
  glVertex2d(215, GLUTwindow_height - 55);
  glVertex2d(215, GLUTwindow_height - 15);

  glEnd();

  double blue_angle = 360.0*(atan2(scene->blueTank.towards.X(), scene->blueTank.towards.Z()) + M_PI)/(2.0*M_PI);
  double blue_tank_x = (scene->blueTank.shape->mesh->bbox.Centroid().X())/15 * 75;
  double blue_tank_z = -(scene->blueTank.shape->mesh->bbox.Centroid().Z())/15 * 75;
  double red_angle = 360.0*(atan2(scene->redTank.towards.X(), scene->redTank.towards.Z()) + M_PI)/(2.0*M_PI);
  double red_tank_x = (scene->redTank.shape->mesh->bbox.Centroid().X())/15 * 75;
  double red_tank_z = -(scene->redTank.shape->mesh->bbox.Centroid().Z())/15 * 75;

  bool original_quality = high_quality;
  high_quality = true;
  LoadMaterial(&map_material);
  high_quality = original_quality;
  glPushMatrix();
  glTranslatef(90, 90, 0);
  if (tank_num == 0) {
      glRotatef(-blue_angle, 0.0f, 0.0f, 1.0f );
  }
  else {
    glRotatef(-red_angle, 0.0f, 0.0f, 1.0f );
  }


  glPushMatrix();
  glTranslatef(blue_tank_x, blue_tank_z ,0);
  glRotatef(blue_angle, 0.0f, 0.0f, 1.0f);
  glColor3f(0.0f, 0.0f, 1.0f);
  glBegin(GL_QUADS);
  glVertex2f(+3,+4);
  glVertex2f(+3,-4);
  glVertex2f(-3,-4);
  glVertex2f(-3,+4);
  glEnd();
  glBegin(GL_QUADS);
  glColor3f(0.0f, 0.0f, 1.0f);
  glVertex2f(+1,+8);
  glVertex2f(+1,0);
  glVertex2f(-1,0);
  glVertex2f(-1,+8);
  glEnd();
  glPopMatrix();

  glPushMatrix();
  glTranslatef(red_tank_x, red_tank_z ,0);
  glRotatef(red_angle, 0.0f, 0.0f, 1.0f);
  glColor3f(1.0f, 0.0f, 0.0f);
  glBegin(GL_QUADS);
  glVertex2f(+3,+4);
  glVertex2f(+3,-4);
  glVertex2f(-3,-4);
  glVertex2f(-3,+4);
  glEnd();
  glBegin(GL_QUADS);
  glColor3f(1.0f, 0.0f, 0.0f);
  glVertex2f(+1,+8);
  glVertex2f(+1,0);
  glVertex2f(-1,0);
  glVertex2f(-1,+8);
  glEnd();
  glPopMatrix();

  glPushMatrix();
  glTranslatef(-90, -90, 0);

  glBegin( GL_QUADS );
  glColor3f(0.6f, 0.6f, 0.6f);
  glTexCoord2f(0.0,1.0); glVertex2f(15,15);
  glTexCoord2f(1.0,1.0); glVertex2f(165,15);
  glTexCoord2f(1.0,0.0); glVertex2f(165,165);
  glTexCoord2f(0.0,0.0); glVertex2f(15,165);
  glEnd();
  glPopMatrix();
  glPopMatrix();
  
	glColor3f(0.0f,0.0f,1.0f);
	glRasterPos2f(GLUTwindow_width/4.0 - 15, GLUTwindow_height - 35);
	char blue [10];
	sprintf(blue, "%d ", scene->blueTank.score);
	char* s = blue;
	while (*s) glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *(s++)); //
	
	glColor3f(0.6f,0.6f,0.6f);
	glRasterPos2f(GLUTwindow_width/4.0, GLUTwindow_height - 35);
	char space [10];
	sprintf(space, "-");
	s = space;
	while (*s) glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *(s++)); //
	
	glColor3f(1.0f,0.0f,0.0f);
	glRasterPos2f(GLUTwindow_width/4.0 + 15, GLUTwindow_height - 35);
	char red [10];
	sprintf(red, "%d", scene->redTank.score);
	s = red;
	while (*s) glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *(s++)); //


  glColor3f(1.0f,1.0f,1.0f);
  glBegin( GL_QUADS );
  glVertex2d(GLUTwindow_width/4.0 - 23, GLUTwindow_height - 15);
  glVertex2d(GLUTwindow_width/4.0 - 23, GLUTwindow_height - 43);
  glVertex2d(GLUTwindow_width/4.0 + 30, GLUTwindow_height - 43);
  glVertex2d(GLUTwindow_width/4.0 + 30, GLUTwindow_height - 15);
 	glEnd();

  glColor3f(0.0f, 0.0f,0.0f);
  glBegin( GL_QUADS );
  glVertex2d(GLUTwindow_width/4.0 - 28, GLUTwindow_height - 10);
  glVertex2d(GLUTwindow_width/4.0 - 28, GLUTwindow_height - 48);
  glVertex2d(GLUTwindow_width/4.0 + 35, GLUTwindow_height - 48);
  glVertex2d(GLUTwindow_width/4.0 + 35, GLUTwindow_height - 10);
 	glEnd();

  glDisable2D();  
}


void key_operations() {
	bool turn = true;
	  if (key_states['w']) {
		  apply_move_forwards_red();
	  }
	  if (key_states['s']) {
		  if (key_states['a']) {
			  apply_turn_red(-.038);
			  turn = false;
		  }
		  if (key_states['d']) {
			  apply_turn_red(.038);
			  turn = false;
		  }
		  apply_move_backwards_red();
	  }
	  if (turn) {
		  if (key_states['a']) {
			apply_turn_red(.038);
		  }
		  if (key_states['d']) {
			apply_turn_red(-.038);
		  }		
	  }
}


void special_key_operations() {
	bool turn = true;
  if (key_special_states[GLUT_KEY_UP]) {
    apply_move_forwards();
  }
  if (key_special_states[GLUT_KEY_DOWN]) {
	  if (key_special_states[GLUT_KEY_LEFT]) {
		  apply_turn(-.038);
		  turn = false;
	  }
	  if (key_special_states[GLUT_KEY_RIGHT]) {
		  apply_turn(.038);
		  turn = false;
	  }
	  apply_move_backwards();
  }
  if (turn) {
	  if (key_special_states[GLUT_KEY_LEFT]) {
		apply_turn(.038);
	  }
	  if (key_special_states[GLUT_KEY_RIGHT]) {
		apply_turn(-.038);
	  }		
  }
}


void DrawSplit1() {
	  
  //Apply key operations
  key_operations();
  special_key_operations();
  if (third_person) {
	  update_camera_rotation();
	  //update_camera2_rotation();
	  update_camera_position();
	  normal_walls();
	  transparent_walls();
  }
  else
	  set_front_camera(0);
  // Initialize OpenGL drawing modes
  glEnable(GL_LIGHTING);
  glEnable(GL_DEPTH_TEST);
  glDisable(GL_BLEND);
  glBlendFunc(GL_ONE, GL_ZERO);
  glDepthMask(true);
  
  // Clear window 
  R3Rgb background = scene->background;
  
  // Load camera
  LoadCamera(&camera);

  
  // Load scene lights
  LoadLights(scene);
  // move bullets
  update_bullets();
  update_particles();
  //Load tanks
  //LoadTanks(tank_position, tank_towards);
  LoadMaterial(scene->blueTank.material);
  DrawShape(scene->blueTank.shape);
  LoadMaterial(scene->redTank.material);
  DrawShape(scene->redTank.shape);
  
  LoadMaterial(scene->greenFlag.material);
  DrawShape(scene->greenFlag.shape);

	
  // Draw scene camera
  DrawCamera(scene);
  
  // Draw scene lights
  DrawLights(scene);

	draw_hud(0);
	
	if (!scene->blueTank.hasFlag) {
		if (intersect_flag(scene, &scene->blueTank)) {
			scene->blueTank.hasFlag = true;
			R3Point center_tank = scene->blueTank.shape->mesh->bbox.Centroid() - scene->blueTank.towards * .35;
			R3Point center_flag = scene->greenFlag.shape->mesh->bbox.Centroid();
			R3Vector difference = center_tank - center_flag;
			scene->greenFlag.shape->mesh->Translate(difference.X(), .3, difference.Z());
		}
	}
	else {
		if (intersect_base(scene, &scene->blueTank)) {
			scene->greenFlag.shape->mesh->Translate(0, -.3, 0);
			scene->blueTank.hasFlag = false;
			reset_flag();
			scene->blueTank.score += 1;
			engine->play2D("../sound/blue_scores.wav", false);
		}
	}
	 
  // Draw scene surfaces
  if (show_faces) {
    glEnable(GL_LIGHTING);
    DrawScene(scene);
  }
  
  // Draw scene edges
  if (show_edges) {
    glDisable(GL_LIGHTING);
    glColor3d(1 - background[0], 1 - background[1], 1 - background[2]);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    DrawScene(scene);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  }
  
  // Quit here so that can save image before exit
  if (quit) {
	  //Enable repeat mode
	  glutSetKeyRepeat(GLUT_KEY_REPEAT_ON);
    GLUTStop();
  }
}

void DrawSplit2(void)
{
	
  //Apply key operations
  key_operations();
  special_key_operations();
  //update_camera_rotation();
  if (third_person_red) {
	  update_camera2_rotation();
	  update_camera2_position();
	  
	  normal_walls();
	  transparent_walls2();
  }
  else
	  set_front_camera(1);
  // Initialize OpenGL drawing modes
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_TEXTURE_2D);
  glEnable(GL_LIGHTING);
  glDisable(GL_BLEND);
  glBlendFunc(GL_ONE, GL_ZERO);
  glDepthMask(true);
  
  // Clear window 
  R3Rgb background = scene->background;
  
  // Load camera
  LoadCamera(&camera2);

  
  // Load scene lights
  LoadLights(scene);
  // move bullets
  update_bullets();
	update_particles();
  //Load tanks
  //LoadTanks(tank_position, tank_towards);
  LoadMaterial(scene->redTank.material);
  DrawShape(scene->redTank.shape);
  LoadMaterial(scene->blueTank.material);
  DrawShape(scene->blueTank.shape);
  
  LoadMaterial(scene->greenFlag.material);
  DrawShape(scene->greenFlag.shape);  

  // Draw scene camera
  DrawCamera(scene);
  
  // Draw scene lights
  DrawLights(scene);

	draw_hud(1);
	
	if (!scene->redTank.hasFlag) {
		if (intersect_flag(scene, &scene->redTank)) {
			scene->redTank.hasFlag = true;
			R3Point center_tank = scene->redTank.shape->mesh->bbox.Centroid() - scene->redTank.towards * .35;
			R3Point center_flag = scene->greenFlag.shape->mesh->bbox.Centroid();
			R3Vector difference = center_tank - center_flag;
			scene->greenFlag.shape->mesh->Translate(difference.X(), .3, difference.Z());
		}
	}
	else {
		if (intersect_base_red(scene, &scene->redTank)) {
			scene->greenFlag.shape->mesh->Translate(0, -.3, 0);
			scene->redTank.hasFlag = false;
			reset_flag();
			scene->redTank.score += 1;
			engine->play2D("../sound/red_scores.wav", false);
		}
	}
  
  // Draw scene surfaces
  if (show_faces) {
    glEnable(GL_LIGHTING);
    DrawScene(scene);
  }
  
  // Draw scene edges
  if (show_edges) {
    glDisable(GL_LIGHTING);
    glColor3d(1 - background[0], 1 - background[1], 1 - background[2]);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    DrawScene(scene);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  }
  
  // Quit here so that can save image before exit
  if (quit) {
	  //Enable repeat mode
	  glutSetKeyRepeat(GLUT_KEY_REPEAT_ON);
    GLUTStop();
  }
}

void GLUTRedraw(void)
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glViewport(0,0, GLUTwindow_width/2, GLUTwindow_height);
  DrawSplit2();
  glViewport(GLUTwindow_width/2, 0, GLUTwindow_width/2, GLUTwindow_height);
  DrawSplit1();
  glutSwapBuffers(); 
}

void GLUTMouse(int button, int state, int x, int y)
{
  // Invert y coordinate
  y = GLUTwindow_height - y;
  
  // Process mouse button event
  if (state == GLUT_DOWN) {
    if (button == GLUT_LEFT_BUTTON) {
    }
    else if (button == GLUT_MIDDLE_BUTTON) {
    }
    else if (button == GLUT_RIGHT_BUTTON) {
    }
  }

  // Remember button state 
  int b = (button == GLUT_LEFT_BUTTON) ? 0 : ((button == GLUT_MIDDLE_BUTTON) ? 1 : 2);
  GLUTbutton[b] = (state == GLUT_DOWN) ? 1 : 0;

  // Remember modifiers 
  GLUTmodifiers = glutGetModifiers();

   // Remember mouse position 
  GLUTmouse[0] = x;
  GLUTmouse[1] = y;

  // Redraw
  glutPostRedisplay();
}



void GLUTSpecial(int key, int x, int y)
{
  // Invert y coordinate
  y = GLUTwindow_height - y;

  // Process keyboard button event 
  switch (key) {
    case GLUT_KEY_LEFT:
      key_special_states[GLUT_KEY_LEFT] = true;
      turning = true;
      
      break;
    case GLUT_KEY_RIGHT:
      key_special_states[GLUT_KEY_RIGHT] = true;
      turning = true;
      break;
    case GLUT_KEY_UP:
      key_special_states[GLUT_KEY_UP] = true;
      break;
    case GLUT_KEY_DOWN:
      key_special_states[GLUT_KEY_DOWN] = true;
      break;
  }

  // Remember mouse position 
  GLUTmouse[0] = x;
  GLUTmouse[1] = y;

  // Remember modifiers 
  GLUTmodifiers = glutGetModifiers();

  // Redraw
  glutPostRedisplay();
}

void GLUTSpecialUp(int key, int x, int y)
{
  // Invert y coordinate
  y = GLUTwindow_height - y;

  // Process keyboard button event 
  switch (key) {
    case GLUT_KEY_LEFT:
      key_special_states[GLUT_KEY_LEFT] = false;
      turning = false;
      break;
    case GLUT_KEY_RIGHT:
      key_special_states[GLUT_KEY_RIGHT] = false;
      turning = false;
			break;
    case GLUT_KEY_UP:
      key_special_states[GLUT_KEY_UP] = false;
      break;
    case GLUT_KEY_DOWN:
      key_special_states[GLUT_KEY_DOWN] = false;
      break;
  }
  // Remember mouse position 
  GLUTmouse[0] = x;
  GLUTmouse[1] = y;

  // Remember modifiers 
  GLUTmodifiers = glutGetModifiers();

  // Redraw
  glutPostRedisplay();
}

void GLUTKeyboard(unsigned char key, int x, int y)
{
  // Invert y coordinate
  y = GLUTwindow_height - y;

  // Process keyboard button event 
  switch (key) {
   case 'w':
     key_states['w'] = true;
     break;
     
   case 'a':
     key_states['a'] = true;
     turning = true;
     break;
     
   case 'd':
     key_states['d'] = true;
     turning = true;
     break;
   
   case 's':
     key_states['s']= true;
     break;
   case 'b':
	   first_person = !first_person;
	   third_person = !third_person;
	   if (third_person) {
		   R3Vector move = R3Vector(0, 8.5,0) - scene->blueTank.towards * 8.5;
		   move.Normalize();
		   camera.eye += move * .2;
	   }
	   break;
   case 'v':
	   first_person_red = !first_person_red;
	   third_person_red = !third_person_red;
	   if (third_person_red) {
		   R3Vector move = R3Vector(0, 8.5,0) - scene->redTank.towards * 8.5;
		   move.Normalize();
		   camera2.eye += move * .2;
	   }
	   break;
    
  case 'f':
    glutFullScreen();
    break;
	case 'r':
		engine->play2D("../audio/music.ogg", true);
		break;
  case 'l': {
    R3Vector blue_translation_vector;
    R3Vector red_translation_vector;
    R3Mesh* bluenew_mesh;
    R3Mesh* blueold_mesh;
    R3Mesh* rednew_mesh;
    R3Mesh* redold_mesh;
    high_quality = !high_quality;
    if (high_quality) {
      bluenew_mesh = high_quality_bluetankmesh;
      blueold_mesh = low_quality_bluetankmesh;
      rednew_mesh = high_quality_redtankmesh;
      redold_mesh = low_quality_redtankmesh;
    }
    else {
      bluenew_mesh = low_quality_bluetankmesh;
      blueold_mesh = high_quality_bluetankmesh;
      rednew_mesh = low_quality_redtankmesh;
      redold_mesh = high_quality_redtankmesh;
    }
    blue_translation_vector = bluenew_mesh->bbox.Centroid() - blueold_mesh->bbox.Centroid();
    red_translation_vector = rednew_mesh->bbox.Centroid() - redold_mesh->bbox.Centroid();

    double blue_rotation_angle = atan2(scene->blueTank.towards.X(), scene->blueTank.towards.Z());
    R3Line bluenewaxis(bluenew_mesh->bbox.Centroid().X(), 1, bluenew_mesh->bbox.Centroid().Z(), bluenew_mesh->bbox.Centroid().X(), 0, bluenew_mesh->bbox.Centroid().Z());
    bluenew_mesh->Rotate(-blue_rotation_angle, bluenewaxis);
    R3Line blueoldaxis(blueold_mesh->bbox.Centroid().X(), 1, blueold_mesh->bbox.Centroid().Z(), blueold_mesh->bbox.Centroid().X(), 0, blueold_mesh->bbox.Centroid().Z());
    blueold_mesh->Rotate(blue_rotation_angle, blueoldaxis);

    double red_rotation_angle = atan2(scene->redTank.towards.X(), scene->redTank.towards.Z());
    R3Line rednewaxis(rednew_mesh->bbox.Centroid().X(), 1, rednew_mesh->bbox.Centroid().Z(), rednew_mesh->bbox.Centroid().X(), 0, rednew_mesh->bbox.Centroid().Z());
    rednew_mesh->Rotate(-red_rotation_angle, rednewaxis);
    R3Line redoldaxis(redold_mesh->bbox.Centroid().X(), 1, redold_mesh->bbox.Centroid().Z(), redold_mesh->bbox.Centroid().X(), 0, redold_mesh->bbox.Centroid().Z());
    redold_mesh->Rotate(red_rotation_angle, redoldaxis);

    scene->blueTank.shape->mesh = bluenew_mesh;
    scene->redTank.shape->mesh = rednew_mesh;
    scene->blueTank.shape->mesh->Translate(-blue_translation_vector.X(), -blue_translation_vector.Y(), -blue_translation_vector.Z());
    scene->redTank.shape->mesh->Translate(-red_translation_vector.X(), -red_translation_vector.Y(), -red_translation_vector.Z());

    break;
  }
  case 27: // ESCAPE
    quit = 1;
    break;
  }

  // Remember mouse position 
  GLUTmouse[0] = x;
  GLUTmouse[1] = y;

  // Remember modifiers 
  GLUTmodifiers = glutGetModifiers();

  // Redraw
  glutPostRedisplay();
}

void GLUTKeyboardUp(unsigned char key, int x, int y)
{
 // Invert y coordinate
  y = GLUTwindow_height - y;

  // Process keyboard button event 
  switch (key) {
	 case 'w':
       key_states['w'] = false;
       break;
       
     case 'a':
       key_states['a'] = false;
       turning = false;
       break;
       
     case 'd':
       key_states['d'] = false;
       turning = false;
       break;
     
     case 's':
       key_states['s'] = false;
       break;
  
    // case 'f':
    //   glutFullScreen();
    //   break;

    case 27: // ESCAPE
      quit = 1;
      break;
    case ' ':
      create_bullet();
	  engine->play2D("../sound/shot.wav", false);
      break;
    case '`':
      create_bullet_red();
	  engine->play2D("../sound/shot.wav", false);
      break;
  }

  // Remember mouse position 
  GLUTmouse[0] = x;
  GLUTmouse[1] = y;

  // Remember modifiers 
  GLUTmodifiers = glutGetModifiers();

  // Redraw
  glutPostRedisplay();
}



void GLUTCommand(int cmd)
{
  // Execute command
  switch (cmd) {
  case DISPLAY_FACE_TOGGLE_COMMAND: show_faces = !show_faces; break;
  case DISPLAY_EDGE_TOGGLE_COMMAND: show_edges = !show_edges; break;
  case DISPLAY_BBOXES_TOGGLE_COMMAND: show_bboxes = !show_bboxes; break;
  case DISPLAY_LIGHTS_TOGGLE_COMMAND: show_lights = !show_lights; break;
  case DISPLAY_CAMERA_TOGGLE_COMMAND: show_camera = !show_camera; break;
  case SAVE_IMAGE_COMMAND: save_image = 1; break;
  case QUIT_COMMAND: quit = 1; break;
  }

  // Mark window for redraw
  glutPostRedisplay();
}



void GLUTCreateMenu(void)
{
  // Display sub-menu
  int display_menu = glutCreateMenu(GLUTCommand);
  glutAddMenuEntry("Faces (F)", DISPLAY_FACE_TOGGLE_COMMAND);
  glutAddMenuEntry("Edges (E)", DISPLAY_EDGE_TOGGLE_COMMAND);
  glutAddMenuEntry("Bounding boxes (B)", DISPLAY_BBOXES_TOGGLE_COMMAND);
  glutAddMenuEntry("Lights (L)", DISPLAY_LIGHTS_TOGGLE_COMMAND);
  glutAddMenuEntry("Camera (C)", DISPLAY_CAMERA_TOGGLE_COMMAND);

  // Main menu
  glutCreateMenu(GLUTCommand);
  glutAddSubMenu("Display", display_menu);
  glutAddMenuEntry("Save Image (F1)", SAVE_IMAGE_COMMAND);
  glutAddMenuEntry("Quit", QUIT_COMMAND);

  // Attach main menu to right mouse button
  glutAttachMenu(GLUT_RIGHT_BUTTON);
}



void GLUTInit(int *argc, char **argv)
{
  // Open window 
  glutInit(argc, argv);
  glutInitWindowPosition(750, 0);
  glutInitWindowSize(GLUTwindow_width, GLUTwindow_height);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH); // | GLUT_STENCIL
  GLUTwindow = glutCreateWindow("Tanks for Playing");

  // Initialize GLUT callback functions 
  glutReshapeFunc(GLUTResize);
  glutDisplayFunc(GLUTRedraw);
  glutKeyboardFunc(GLUTKeyboard);
  glutKeyboardUpFunc(GLUTKeyboardUp);
  glutSpecialFunc(GLUTSpecial);
  glutSpecialUpFunc(GLUTSpecialUp);
  glutMouseFunc(GLUTMouse);
  glutMotionFunc(GLUTMotion);

  // Initialize graphics modes 
  glEnable(GL_NORMALIZE);
  glEnable(GL_LIGHTING);
  glEnable(GL_DEPTH_TEST);
  glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE);
 
  // Create menus
  GLUTCreateMenu();
  
  //Enable repeat mode
  glutSetKeyRepeat(GLUT_KEY_REPEAT_OFF);
}




////////////////////////////////////////////////////////////
// SCENE READING
////////////////////////////////////////////////////////////


R3Scene *
ReadScene(const char *filename)
{
  // Allocate scene
  R3Scene *scene = new R3Scene();
  if (!scene) {
    fprintf(stderr, "Unable to allocate scene\n");
    return NULL;
  }

  // Read file
  if (!scene->Read(filename)) {
    fprintf(stderr, "Unable to read scene from %s\n", filename);
    return NULL;
  }
  //Create new camera
  camera = scene->camera;
	R3Vector towards (0, -1, 0);
	R3Vector right (1, 0, 0);
	R3Vector up (0, 0, 1);
	R3Point eye (0, 100, 0);
	R3Vector xz (0, -1, 0);
	first_person = 0;
	third_person = 1;
	first_person_red = 0;
	third_person_red = 1;
	if (first_person) {
		towards.Reset(1, 0, 0);
		right.Reset(0, 0, 1);
		up.Reset(0, 1, 0);
		eye.Reset(-13.5, .5, 13.5);
		xz.Reset(1, 0, 0);
	}
	if (third_person) {
		  // set initial camera
		  //eye.Reset(-22, 8.5, 13.5);
		eye.Reset(0, 14.5, 0);
		  towards = scene->blueTank.shape->mesh->bbox.Centroid() - eye + R3Vector(0, 1.8, 0);
		  towards.Normalize();
		  right.Reset(0, 0, 1);
		  xz.Reset(1, 0, 0);
		  up = right;
		  up.Cross(towards);
		  up.Normalize();
	}
	
	camera.xz = xz;
  camera.towards = towards;
  camera.right = right;
  camera.up = up;
  camera.eye = eye;
	  
  camera2 = scene->camera2;
	if (first_person_red) {
		towards.Reset(1, 0, 0);
		right.Reset(0, 0, 1);
		up.Reset(0, 1, 0);
		eye.Reset(-13.5, .5, -13.5);
		xz.Reset(1, 0, 0);
	}
	if (third_person_red) {
		  // set initial camera
			eye.Reset(0, 14.5, 0);
		  //eye.Reset(-22, 8.5, -13.5);
		  towards = scene->redTank.shape->mesh->bbox.Centroid() - eye + R3Vector(0, 1.8, 0);
		  towards.Normalize();
		  right.Reset(0, 0, 1);
		  xz.Reset(1, 0, 0);
		  up = right;
		  up.Cross(towards);
		  up.Normalize();
	}
	
	camera2.xz = xz;
	camera2.towards = towards;
	camera2.right = right;
	camera2.up = up;
	camera2.eye = eye;
	camera2.xfov = camera.xfov;
	camera2.yfov = camera.yfov;
	camera2.neardist = camera.neardist;
	camera2.fardist = camera.fardist;
  // camera.right.Print();
  // puts("");
  // camera.up.Print();
  // puts("");
  // camera.towards.Print();
  // puts("");
  // camera.eye.Print();

  // Return scene
  return scene;
}



////////////////////////////////////////////////////////////
// PROGRAM ARGUMENT PARSING
////////////////////////////////////////////////////////////

int 
ParseArgs(int argc, char **argv)
{
  // Innocent until proven guilty
  int print_usage = 0;

  // Parse arguments
  argc--; argv++;
  while (argc > 0) {
    if ((*argv)[0] == '-') {
      if (!strcmp(*argv, "-help")) { print_usage = 1; }
      else if (!strcmp(*argv, "-exit_immediately")) { quit = 1; }
      else if (!strcmp(*argv, "-output_image")) { argc--; argv++; output_image_name = *argv; }
      else { fprintf(stderr, "Invalid program argument: %s", *argv); exit(1); }
      argv++; argc--;
    }
    else {
      if (!input_scene_name) input_scene_name = *argv;
      else { fprintf(stderr, "Invalid program argument: %s", *argv); exit(1); }
      argv++; argc--;
    }
  }

  // Check input_scene_name
  if (!input_scene_name || print_usage) {
    printf("Usage: rayview <input.scn> [-maxdepth <int>] [-v]\n");
    return 0;
  }

  // Return OK status 
  return 1;
}



////////////////////////////////////////////////////////////
// MAIN
////////////////////////////////////////////////////////////

int 
main(int argc, char **argv)
{
  // Initialize GLUT
  GLUTInit(&argc, argv);

  // Parse program arguments
  if (!ParseArgs(argc, argv)) exit(1);

  // Read scene
  scene = ReadScene(input_scene_name);
  if (!scene) exit(-1);
	
	reset_flag();

  // Load map texture
  map_texture_image= R2Image("../textures/map.jpg");
  map_material.texture = &map_texture_image;

  high_quality_bluetankmesh = scene->blueTank.shape->mesh;
  low_quality_bluetankmesh = scene->blueTankSimple;
  high_quality_redtankmesh = scene->redTank.shape->mesh;
  low_quality_redtankmesh = scene->redTankSimple;

  engine->play2D("../sound/engine.wav", true);

  // Run GLUT interface
  GLUTMainLoop();

  // Return success 
  return 0;
}









