// Include file for the R3 scene stuff

#define R3Rgb R2Pixel

#include "R3/R3.h"

// Constant definitions

typedef enum {
  R3_BOX_SHAPE,
  R3_SPHERE_SHAPE,
  R3_CYLINDER_SHAPE,
  R3_CONE_SHAPE,
  R3_MESH_SHAPE,
  R3_SEGMENT_SHAPE,
  R3_NUM_SHAPE_TYPES
} R3ShapeType;

typedef enum {
  R3_DIRECTIONAL_LIGHT,
  R3_POINT_LIGHT,
  R3_SPOT_LIGHT,
  R3_AREA_LIGHT,
  R3_NUM_LIGHT_TYPES
} R3LightType;

// Scene element definitions

struct R3Shape {
  R3ShapeType type;
  R3Box *box;
  R3Sphere *sphere;
  R3Cylinder *cylinder;
  R3Cone *cone;
  R3Mesh *mesh;
  R3Segment *segment;
};  

struct R3Material {
  R3Rgb ka;
  R3Rgb kd;
  R3Rgb ks;
  R3Rgb kt;
  R3Rgb emission;
  double shininess;
  double indexofrefraction;
  R2Image *texture;
  int texture_index;
  int id;
};
struct Tank {
	R3Shape *shape;
	//R3Matrix transformation;
	R3Vector velocity;
	R3Vector towards;
	R3Vector up;
	R3Vector right;
	bool canShoot;
	double shotSpeed;
  double hasFlag;
	double rotated; // total amount rotated, used to reset the tank when hit
  bool turn_left;
  bool turn_right;
  int score;
  int health;
  double bulletDelay;
	R3Material *material;
  vector<R3Point> corners;
};
struct Bullet {
	R3Shape *shape;
	R3Vector towards;
	R3Material *material;
	int color;
	double lifetime; // add some constant to this with each redraw and if ever bigger than threshold, delete
};

struct Flag {
	R3Shape *shape;
	//R3Matrix transformation;
	R3Vector velocity;
	R3Vector towards;
	R3Vector up;
	R3Vector right;
	R3Material *material;
};

struct R3Light {
  R3LightType type;
  R3Point position;
  R3Vector direction;
  double radius;
  R3Rgb color;
  double constant_attenuation;
  double linear_attenuation;
  double quadratic_attenuation;
  double angle_attenuation;
  double angle_cutoff;
};

struct R3Particle {
	R3Shape *shape;
  R3Point position;
  R3Vector velocity;
  double mass;
  bool fixed;
  double drag;
  double elasticity;
  double lifetime;
  int color;
  R3Material *material;
};

struct R3Camera {
  R3Point eye;
  R3Vector towards;
  R3Vector right;
  R3Vector up;
  R3Vector xz;
  double xfov, yfov;
  double neardist, fardist;
};


struct R3Node {
  struct R3Node *parent;
  vector<struct R3Node *> children;
  R3Shape *shape;
  R3Matrix transformation;
  R3Material *material;
  R3Box bbox;
};

struct R3Intersection {
	bool hit;
	R3Node *node;
	R3Point position;
	R3Vector normal;
	double t;
};


// Scene graph definition

struct R3Scene {
 public:
  // Constructor functions
  R3Scene(void);

  // Access functions
  R3Node *Root(void) const;
  int NLights(void) const;
  R3Light *Light(int k) const;
  R3Camera& Camera(void);
  R3Box& BBox(void);

  // I/O functions
  int Read(const char *filename, R3Node *root = NULL);

 public:
  R3Node *root;
  vector<R3Light *> lights;
  R3Camera camera;
  R3Camera camera2;
  R3Box bbox;
  R3Rgb background;
  R3Rgb ambient;
  Tank blueTank;
  Tank redTank;
  R3Mesh* blueTankSimple;
  R3Mesh* redTankSimple;
  Flag greenFlag;
  double cameraUpdate; // these are used to move the camera slower than tank
  double cameraUpdate2;
  double cameraPosition;
  double cameraPosition2;
  vector<Bullet *> bullets;
  vector<R3Node *> transparentWalls;
  vector<R3Particle *> particles;
  R3Material *bulletMaterial;
  R3Vector gravity;
};



// Inline functions 

inline R3Node *R3Scene::
Root(void) const
{
  // Return root node
  return root;
}



inline int R3Scene::
NLights(void) const
{
  // Return number of lights
  return lights.size();
}



inline R3Light *R3Scene::
Light(int k) const
{
  // Return kth light
  return lights[k];
}



inline R3Camera& R3Scene::
Camera(void) 
{
  // Return camera
  return camera;
}



inline R3Box& R3Scene::
BBox(void) 
{
  // Return bounding box 
  return bbox;
}



