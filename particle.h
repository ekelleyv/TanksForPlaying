// Include file for the particle system



// Particle system integration types

enum {
  EULER_INTEGRATION,
  MIDPOINT_INTEGRATION,
  ADAPTIVE_STEP_SIZE_INTEGRATION
};



// Particle system functions
R3Vector GetForce(R3Scene *scene, R3Particle p, double current_time);
void UpdateParticles(R3Scene *scene, double current_time, double delta_time, int integration_type);
void GenerateParticles(R3Scene *scene, double current_time, double delta_time);
void RenderParticles(R3Scene *scene, double current_time, double delta_time);
void printVector(R3Vector v);
void DeleteParticle(R3Scene *scene, int index);
// main function
R2Image *RenderImage(R3Scene *scene, int width, int height, int max_depth, 
  int num_primary_rays_per_pixel, int num_distributed_rays_per_intersection, bool transmission);

// construct the rays evenly spaced through image
R3Ray ConstructRayThroughPixel(R3Camera camera, int i, int j, int width, int height);

// checking intersections
R3Intersection *ComputeIntersectionNode(R3Scene *scene, R3Ray *ray, R3Node *node, double minT);
R3Intersection *ComputeIntersection(R3Scene *scene, R3Ray *ray);

// computing radiance
R3Rgb ComputeRadiance(R3Scene *scene, R3Ray *ray, int max_depth, double muI, bool transmission);
R3Rgb ComputeRadiance(R3Scene *scene, R3Ray *ray, R3Intersection *intersection, int max_depth, double muI, bool transmission);
R3Intersection *checkShadowRay(R3Scene *scene, R3Point originalIntersection, R3Vector L, double d);

// intersection functions
R3Intersection *IntersectSphere(R3Ray *ray, R3Node *node);
R3Intersection *IntersectMesh(R3Ray *ray, R3Node *node);
R3Intersection *IntersectPolygon(R3Ray *ray, R3MeshFace *face);
R3Intersection *IntersectCylinder(R3Ray *ray, R3Node *node);
R3Intersection *IntersectCone(R3Ray *ray, R3Node *node);
R3Intersection *IntersectBox(R3Ray *ray, R3Node *node);
R3Intersection *IntersectBoundingBox(R3Ray *ray, R3Box box);
double IntersectPlane(R3Ray *ray, R3Plane p);
