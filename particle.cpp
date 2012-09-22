// Source file for the particle system



// Include files

#include "R2/R2.h"
#include "R3/R3.h"
#include "R3Scene.h"
#include "particle.h"
#include "raytrace.h"
#include <float.h>
using namespace std;
#ifdef _WIN32
#   include <windows.h>
#else
#   include <sys/time.h>
#endif




////////////////////////////////////////////////////////////
// Random Number Generator
////////////////////////////////////////////////////////////

double RandomNumber(void)
{
#ifdef _WIN32
  // Seed random number generator
  static int first = 1;
  if (first) {
    srand(GetTickCount());
    first = 0;
  }

  // Return random number
  int r1 = rand();
  double r2 = ((double) rand()) / ((double) RAND_MAX);
  return (r1 + r2) / ((double) RAND_MAX);
#else 
  // Seed random number generator
  static int first = 1;
  if (first) {
    struct timeval timevalue;
    gettimeofday(&timevalue, 0);
    srand48(timevalue.tv_usec);
    first = 0;
  }

  // Return random number
  return drand48();
#endif
}



////////////////////////////////////////////////////////////
// Generating Particles
////////////////////////////////////////////////////////////

void GenerateParticles(R3Scene *scene, double current_time, double delta_time)
{
	// Generate new particles for every source
	for (int i = 0; i < scene->NParticleSources(); i++) {
		R3ParticleSource *p = scene->ParticleSource(i);
		double toOutput = p->rate * delta_time;
		int iToOutput = (int) toOutput;
		double prob = toOutput - iToOutput;
		// use probability
		double randProb = RandomNumber();
		if (randProb < prob) {
			iToOutput += 1;
		}
		fprintf(stderr, "%d %f %f %f\n", iToOutput, p->rate * delta_time, prob, randProb);
		//else
		R3Particle *particle = new R3Particle();
		for (int j = 0; j < iToOutput; j++) {
			if (p->shape->type == R3_SPHERE_SHAPE) {
				double z = 2*RandomNumber() - 1;// should be between [-1,1]
				double t = 2*M_PI*RandomNumber();
				double r = sqrt(1-z*z);
				r *= p->shape->sphere->Radius();
				double x = r*cos(t);
				double y = r*sin(t);
				z *= p->shape->sphere->Radius();
				R3Point center = p->shape->sphere->Center();
				x += center.X();
				y += center.Y();
				z += center.Z();
				R3Point randomP(x, y, z);
				R3Vector v = randomP - center;
				v.Normalize();
				particle->velocity = v;
				particle->position = randomP;
			}
			else if (p->shape->type == R3_CIRCLE_SHAPE) {
				R3Vector circleN = p->shape->circle->Normal();
				R3Vector tangent(circleN.Z() * -1.0, circleN.Z() * -1.0, circleN.X() + circleN.Y());
				double angle = 2*M_PI*RandomNumber();
				tangent.Normalize();
				tangent.Rotate(circleN, angle);
				double randomFact = RandomNumber() * p->shape->circle->Radius();
				R3Point center = p->shape->circle->Center();
				R3Point randomP = tangent.Point()*randomFact;
				randomP += center;
				R3Vector v = circleN;
				particle->velocity = v;
				particle->position = randomP;
			}
			else if (p->shape->type == R3_SEGMENT_SHAPE) {
				
				R3Vector segV= p->shape->segment->Vector();
				R3Vector N(segV.Z() * -1.0, segV.Z() * -1.0, segV.X() + segV.Y());
				double angle = 2*M_PI*RandomNumber();
				N.Normalize();
				N.Rotate(segV, angle);
				double t = RandomNumber();
				R3Point randomP = t*p->shape->segment->Start() + (1-t) * p->shape->segment->End();
				particle->velocity = N;
				particle->position = randomP;
			}
			
			else
				continue;
			particle->mass = p->mass;
			particle->drag = p->drag;
			particle->elasticity = p->elasticity;
			particle->material = p->material;
			particle->start = current_time;
			vector<struct R3ParticleSpring *> *s = new vector<struct R3ParticleSpring *>();
			particle->springs = *s;
			particle->lifetime = p->lifetime;

			
			// rotate a random angle
			R3Vector A(particle->velocity.Z()*-1.0, particle->velocity.Z()*-1.0, particle->velocity.X() + particle->velocity.Y());
			double t1 = RandomNumber() * 2 * M_PI;
			double t2 = RandomNumber() * sin(p->angle_cutoff);
			A.Rotate(particle->velocity, t1);
			R3Vector axis = A;
			axis.Cross(particle->velocity);
			A.Rotate(axis, acos(t2));
			A.Normalize();
			particle->velocity = A*p->velocity;
			scene->particles.push_back(particle);
			
		}
		fprintf(stderr, "%d %f\n", scene->NParticles(), current_time);
	}

}

////////////////////////////////////////////////////////////
// Updating Particles
////////////////////////////////////////////////////////////

void UpdateParticles(R3Scene *scene, double current_time, double delta_time, int integration_type)
{
	// create a vector to update new particle positions
	vector<R3Point> newPositions;
	vector<R3Vector> newVelocities;
	for (int i = 0; i < scene->NParticles(); i++) {
		R3Point point(0,0,0);
		R3Vector vector(0, 0, 0);
		newPositions.push_back(point);
		newVelocities.push_back(vector);
	}
	int oldNumParticles = scene->NParticles();
	
	// Find new position and velocity position for every particle
	for (int i = oldNumParticles -1; i >= 0 ; i--) {

		bool deleted = false;
		R3Particle *p = scene->Particle(i);
		R3Point pos = p->position;
		R3Vector v = p->velocity;
		double mass = p->mass;
		R3Vector force(0, 0, 0);
		R3Vector blank(0, 0, 0);
		// account for drag
		double k = -1.0*p->drag;
		if (v != blank)
			force += v * k;


		if (scene->gravity != blank)
			force += scene->gravity * mass;

		for (int j = 0; j < scene->NParticleSinks(); j++) {
			// sphere sink
			if (scene->ParticleSink(j)->shape->type == R3_SPHERE_SHAPE) {
				R3ParticleSink *sink = scene->ParticleSink(j);
				R3Vector to = sink->shape->sphere->Center() - pos;
				double d = to.Length();
				to.Normalize();
				d -= sink->shape->sphere->Radius();
				if (d <= 0) {
					deleted = true;
				}
				double quad = sink->constant_attenuation + d * sink->linear_attenuation + d * d * sink->quadratic_attenuation;
				if (quad == 0) continue;
				R3Vector sinkForce = to * sink->intensity / quad;
				force += sinkForce;
			}
			// box shape sink
			else if (scene->ParticleSink(j)->shape->type == R3_BOX_SHAPE) {
				R3ParticleSink *sink = scene->ParticleSink(j);
				R3Box *b = sink->shape->box;
				R3Vector to = b->ClosestPoint(pos) - pos;
				double d = to.Length();
				to.Normalize(); // NOT SURE WHAT SIZE THIS SHOULD BE? NEGLIGIBLE? NORMALIZED?
				if (pos.X() >= b->XMin() && pos.X() <= b->XMax() && pos.Y() >= b->YMin() && pos.Y() <= b->YMax() && pos.Z() >= b->ZMin() && pos.Z() <= b->ZMax()) {
					deleted = true;
				}
				if (d <= 0)
					deleted = true;
				double quad = sink->constant_attenuation + d * sink->linear_attenuation + d * d * sink->quadratic_attenuation;
				if (quad == 0) continue;
				R3Vector sinkForce = to * sink->intensity / quad;
				force += sinkForce;
			}
			// segment sink
			else if (scene->ParticleSink(j)->shape->type == R3_SEGMENT_SHAPE) {
				R3ParticleSink *sink = scene->ParticleSink(j);
				R3Segment *seg = sink->shape->segment;
				R3Vector end = seg->End()-pos;
				end /= end.Length() * end.Length();
				double t = (seg->Start() - pos).Dot(end);
				if (t < 0) t = 0;
				else if (t > 1) t = 1;
				R3Point linePoint = (1-t) * seg->Start() + t * seg->End();
				R3Vector to = linePoint - pos;
				double d = to.Length();
				if (d == 0)
					deleted = true;
				to.Normalize();
				
				double quad = sink->constant_attenuation + d * sink->linear_attenuation + d * d * sink->quadratic_attenuation;
				if (quad == 0) continue;
				R3Vector sinkForce = to * sink->intensity / quad;
				force += sinkForce;
			}
		}
		
		// calculate attraction force
		for (int j = 0; j < oldNumParticles; j++) {
			// don't calculate on self
			if (j != i) {
				R3Particle *neighborP = scene->Particle(j);
				R3Vector attract = neighborP->position - p->position;
				double d = attract.Length();
				if (d == 0) continue;
				attract.Normalize();
				attract *= .0000000000667428 * p->mass * neighborP->mass / (d * d);
				force += attract;
			}
		}

		// calculate spring force
		for (unsigned int j = 0; j < p->springs.size(); j++) {
			R3ParticleSpring *spring = p->springs[j];
			R3Particle *pSpring = spring->particles[0];
			R3Particle *qSpring = spring->particles[1];
			if (pSpring->position != p->position) {
				R3Particle *temp = qSpring;
				qSpring = pSpring;
				pSpring = temp;
			}
			R3Vector D = qSpring->position - pSpring->position;
			double d = D.Length();
			D.Normalize();
			R3Vector qV = qSpring->velocity;
			R3Vector pV = pSpring->velocity;
			double ksFactor = spring->ks*(d-spring->rest_length);
			double kdFactor = spring->kd*(qV - pV).Dot(D);
			force += (ksFactor + kdFactor) * D;			
		}


		double time = current_time - p->start;
		// check lifetime
		if (time > p->lifetime && p->lifetime > 0) {
			deleted = true;
		}
		if (deleted) {
			scene->particles[i]->mass = -1.0;
		}
		else {
			
			R3Point newPos = pos + delta_time*v.Point();
			
			R3Ray *ray = new R3Ray(pos, v);
			double newT = ray->T(newPos);
			// check if collision is going to occur
			R3Intersection *intersect = ComputeIntersection(scene, ray);
			double collisionT = ray->T(intersect->position);

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
			}
			
			R3Vector newV = v + delta_time*force/mass;
			if (!p->fixed) {
				newPositions[i] = newPos;
				newVelocities[i] = newV;
			}
			else {
				newPositions[i] = pos;
			}
		}	
	}

	// now update all of the partcles
	for (int i = oldNumParticles -1; i >=0; i--) {
		if (scene->particles[i]->mass == -1.0)
			DeleteParticle(scene, i);
		else {
			scene->particles[i]->position = newPositions[i];
			scene->particles[i]->velocity = newVelocities[i];
		}
	}


}

void DeleteParticle(R3Scene *scene, int index) {
	// Remove particle from list
	if (index < scene->NParticles()) {
		R3Particle *p = scene->particles[index];
		scene->particles[index] = scene->particles.back();
		//vertices[i]->id = i;
		scene->particles.pop_back();
	
		// Delete particle
		//delete p;
	}
}

////////////////////////////////////////////////////////////
// Rendering Particles
////////////////////////////////////////////////////////////

void RenderParticles(R3Scene *scene, double current_time, double delta_time)
{
  // Draw every particle

  // REPLACE CODE HERE
  glDisable(GL_LIGHTING);
  glPointSize(5);
  glBegin(GL_POINTS);
  for (int i = 0; i < scene->NParticles(); i++) {
    R3Particle *particle = scene->Particle(i);
    double life = particle->lifetime;

    double time = 1 - (current_time - particle->start)/life;
    if (life == 0)
    	time = 1;
    glColor3d(particle->material->kd[0] * time, particle->material->kd[1] * time, particle->material->kd[2]* time);
    const R3Point& position = particle->position;
    glVertex3d(position[0], position[1], position[2]);
  }   
  glEnd();
}


void printVector(R3Vector v) {
	fprintf(stderr, "(%f, %f, %f)\n", v.X(), v.Y(), v.Z());
}

// Source file for raytracing code

////////////////////////////////////////////////////////////////////////
// Create image from scene
//
// This is the main ray tracing function called from raypro
// 
// "width" and "height" indicate the size of the ray traced image
//   (keep these small during debugging to speed up your code development cycle)
//
// "max_depth" indicates the maximum number of secondary reflections/transmissions to trace for any ray
//   (i.e., stop tracing a ray if it has already been reflected max_depth times -- 
//   0 means direct illumination, 1 means one bounce, etc.)
//
// "num_primary_rays_per_pixel" indicates the number of random rays to generate within 
//   each pixel during antialiasing.  This argument can be ignored if antialiasing is not implemented.
//
// "num_distributed_rays_per_intersection" indicates the number of secondary rays to generate
//   for each surface intersection if distributed ray tracing is implemented.  
//   It can be ignored otherwise.
// 
////////////////////////////////////////////////////////////////////////

R2Image *RenderImage(R3Scene *scene, int width, int height, int max_depth,
  int num_primary_rays_per_pixel, int num_distributed_rays_per_intersection, bool transmission)
{
  // Allocate  image
  R2Image *image = new R2Image(width, height);
  if (!image) {
    fprintf(stderr, "Unable to allocate image\n");
    return NULL;
  }
  for (int i = 0; i < width; i++) {
	  for (int j = 0; j < height; j++) {
		  
		  // construct rays
		  R3Ray ray = ConstructRayThroughPixel(scene->camera, i, j, width, height);
		  // get radiance
		  R3Rgb radiance = ComputeRadiance(scene, &ray, max_depth, 1, transmission);
		  // change pixel of image to found radiance
		  image->SetPixel(i, j, radiance);			  
	  }
  }

  // Return image
  return image;
}

R3Ray ConstructRayThroughPixel(R3Camera camera, int i, int j, int width, int height)
{
	R3Vector up = camera.up;

	R3Vector right = camera.right;
	R3Vector towards = camera.towards;
	double xfov = camera.xfov;
	double yfov = camera.yfov;
	// size of the camera vision
	R3Vector xdeviance = tan(xfov) * right;
	R3Vector ydeviance = tan(yfov) * up;

	// find four corners of image
	R3Point topR = camera.eye + towards + xdeviance + ydeviance;
	R3Point bottomR = camera.eye + towards + xdeviance - ydeviance;
	R3Point topL = camera.eye + towards - xdeviance + ydeviance;
	R3Point bottomL = camera.eye + towards - xdeviance - ydeviance;
	
	double xdist = ((double)i + .5)/width;
	double ydist = ((double)j + .5)/height;
	// interpolate between horizontal pairs (if xdist is 0, take the left points)
	R3Point topP = xdist * topR + (1-xdist) * topL;
	R3Point bottomP = xdist * bottomR + (1-xdist) * bottomL;
	
	// interpolate between vertical pairs (if ydist is 0, take the bottom point)
	R3Point P = ydist * topP + (1-ydist) * bottomP;
	
	R3Vector vect = (P-camera.eye);
	vect.Normalize();
	R3Ray ret(camera.eye, vect);
	return ret;
}

R3Rgb ComputeRadiance(R3Scene *scene, R3Ray *ray, int max_depth, double muI, bool transmission)
{
	//R3Rgb white(1, 1, 1,1);
	R3Intersection *intersection = ComputeIntersection(scene, ray);
	if (intersection->hit)
		return ComputeRadiance(scene, ray, intersection, max_depth, muI, transmission);
		//return white;
	return scene->background; // return the background color if no intersection found
}

R3Intersection *ComputeIntersectionNode(R3Scene *scene, R3Ray *ray, R3Node *node, double minT)
{
	
	R3Intersection *intersect = new R3Intersection();
	intersect->hit = false;
	intersect->t = minT;

	// need to transform ray
	R3Matrix transform = node->transformation;
	ray->InverseTransform(transform);
	



	// now check parent node
	if (node->shape != NULL) {
		// sphere intersection
		if (node->shape->type == R3_SPHERE_SHAPE) {
			R3Intersection *parentI = IntersectSphere(ray, node);
			if (parentI->hit) {
				if (parentI->t < intersect->t)
					intersect = parentI;
				return intersect;
			}
		}
		// mesh intersection
		else if (node->shape->type == R3_MESH_SHAPE) {
			R3Intersection *parentI= IntersectMesh(ray, node);
			if (parentI->hit) {
				if (parentI->t < intersect->t)
					intersect = parentI;
				return intersect;
			}
		}
		// cylinder intersection
		else if (node->shape->type == R3_CYLINDER_SHAPE) {
			R3Intersection *parentI= IntersectCylinder(ray, node);
			if (parentI->hit) {
				if (parentI->t < intersect->t)
					intersect = parentI;
				//fprintf(stderr, "intersected cylinder\n");
				return intersect;
			}
		}
		// cone intersection
		else if (node->shape->type == R3_CONE_SHAPE) {
			R3Intersection *parentI = IntersectCone(ray, node);
			if (parentI->hit) {
				if (parentI->t < intersect->t)
					intersect = parentI;
				return intersect;
			}
		}
		// box intersection
		else if (node->shape->type == R3_BOX_SHAPE) {
			R3Intersection *parentI = IntersectBox(ray, node);
			if (parentI->hit) {
				if (parentI->t < intersect->t)
					intersect = parentI;
				return intersect;
			}
		}
	}
	
	// check for intersection in all children
	for (unsigned int i = 0; i < node->children.size(); i++) {
		R3Intersection *childI = ComputeIntersectionNode(scene, ray, node->children[i], DBL_MAX);
		if (childI->hit) {
			if (childI->t < intersect->t)
				intersect = childI;
		}
	}
	
	// transform ray back
	ray->Transform(transform);
	if (intersect->hit) {
		// transform intersection back

		intersect->normal.Transform(transform);
		intersect->position.Transform(transform);
		R3Vector vect = intersect->position - ray->Start();
		intersect->t = vect.Point().X() / ray->Vector().X();
	}
	
	return intersect;
}
// search through all nodes in the scene for intersection with ray
R3Intersection *ComputeIntersection(R3Scene *scene, R3Ray *ray)
{	
	R3Node *node = scene->root;
	R3Intersection *smallestIntersect = new R3Intersection();
	smallestIntersect->hit = false;
	smallestIntersect->t = DBL_MAX;
	// check all children nodes of the root
	for (unsigned int i = 0; i < node->children.size(); i++) {
		R3Intersection *intersect = ComputeIntersectionNode(scene, ray, node->children[i], DBL_MAX);
		// check if found the smallest intersection yet
		if (intersect->hit) {
			if (intersect->t < smallestIntersect->t) {
				smallestIntersect = intersect;
			}
		}
	}
	return smallestIntersect;
}
R3Intersection *checkShadowRay(R3Scene *scene, R3Point originalIntersection, R3Vector L, double d) {
	R3Point shadowPoint = originalIntersection + L*.001;
	R3Ray *shadowRay = new R3Ray(shadowPoint, L);
	R3Intersection *shadowI = ComputeIntersection(scene, shadowRay);
	if (shadowI->hit) {
		double shadowD = (shadowI->position - originalIntersection).Length();
		if (shadowD >= d) {
			shadowI->hit = false;
		}
	}
	return shadowI;
}
R3Rgb ComputeRadiance(R3Scene *scene, R3Ray *ray, R3Intersection *intersection, int max_depth, double muI, bool transmission)
{
	R3Material *m = intersection->node->material;
	
	R3Vector N = intersection->normal;
	N.Normalize();
	R3Rgb color = scene->ambient*m->ka + m->emission;
	
	R3Rgb Ir(0, 0, 0, 1);
	R3Rgb It(0, 0, 0, 1);
	R3Vector V = ray->Vector();
	V.Normalize();
	if (max_depth > 0) {
		// find reflection contribution
		if (m->ks.Red() != 0 || m->ks.Green() != 0 || m->ks.Blue() != 0) {
			// reflect off surface	
			R3Vector reflectVector = V- 2*V.Dot(N) * N;
			reflectVector.Normalize();
			R3Ray *reflectRay = new R3Ray(intersection->position + reflectVector*.001, reflectVector);
			Ir = ComputeRadiance(scene, reflectRay, max_depth - 1, 1, transmission);
		}
		
		// find transmission contribution
		if (transmission) {
			if (m->kt.Red() != 0 || m->kt.Green() != 0 || m->kt.Blue() != 0) {
				// reflect off surface	
				R3Vector tVector = V;
				tVector.Normalize();
				R3Ray *tRay = new R3Ray(intersection->position + tVector*.001, tVector);
				It = ComputeRadiance(scene, tRay, max_depth -1, 1, transmission);
			}
		}
		// find refraction contribution
		else {
			if (m->kt.Red() != 0 || m->kt.Green() != 0 || m->kt.Blue() != 0) {
				// reflect off surface	
				R3Vector tVector = V;
				double mI = muI;
				
				double muR = m->indexofrefraction;
				if (max_depth == 1)
					muR = 1.0;
				double mRatio = mI/muR;
				
				double nFactor = 1 - mRatio*mRatio*(1-(tVector.Dot(N)*tVector.Dot(N)));
					
					R3Vector refractV = mRatio * (tVector - N*tVector.Dot(N)) - N * sqrt(nFactor);
					R3Ray *refractRay = new R3Ray(intersection->position + refractV*.001, refractV);
					
					It = ComputeRadiance(scene, refractRay, max_depth -1, muR, transmission);
			}
		}
	}
	R3Rgb reflectance = m->ks * Ir;
	color += reflectance;
	
	R3Rgb trans = m->kt * It;
	color += trans;

	V.Flip();

	if (intersection->hit == false) return color;
	for (unsigned int i = 0; i < scene->lights.size(); i++) {
		R3Light *light = scene->Light(i);
		R3Rgb lightColor = light->color;
		
		// set up vectors
		R3Vector Lto = light->position - intersection->position; // vector to light
		R3Vector Lfrom = intersection->position - light->position; // vector from light
		double d = Lfrom.Length();
		Lfrom.Normalize();
		Lto.Normalize();

		
		// compute IL for each type of light, handling attenuation and shadows
		if (light->type == R3_POINT_LIGHT) {
			R3Intersection *shadowI = checkShadowRay(scene, intersection->position, Lto, d);
			if (shadowI->hit)
				continue;

			// calculate attenuation factor
			double factor = light->constant_attenuation + light->linear_attenuation * d + light->quadratic_attenuation * d * d;
			
			// don't divide by zero
			if (factor <= 0) {
				continue;
			}
			lightColor /= factor;
		}
		else if (light->type == R3_SPOT_LIGHT) {
			R3Intersection *shadowI = checkShadowRay(scene, intersection->position, Lto, d);
			if (shadowI->hit)
				continue;
			
			R3Vector direction = light->direction;
			direction.Normalize();
			double theta = acos(Lfrom.Dot(direction));
			double factor = light->constant_attenuation + light->linear_attenuation * d + light->quadratic_attenuation * d * d;
			// no negative colors
			if (factor/cos(theta) <= 0)
				continue;
			if (theta <= light->angle_cutoff && factor > 0)
				lightColor = light->color  * pow(cos(theta), light->angle_attenuation) / factor;
			else 
				continue;
		}
		else if (light->type == R3_DIRECTIONAL_LIGHT) {
			// change light vectors to be direction not position based
			Lfrom = light->direction;
			Lfrom.Normalize();
			Lto = Lfrom;
			Lto.Flip();
			
			lightColor = light->color;
			R3Intersection *shadowI = checkShadowRay(scene, intersection->position, Lto, DBL_MAX);
			if (shadowI->hit) {
				continue;
			}			
		}
		else if (light->type == R3_AREA_LIGHT) {
			R3Vector direction = light->direction;
			direction.Normalize();
			int counter = 0;
			for (int i = 0; i < 16; i++) {
				R3Vector inPlane = direction;
				double randX = 2.0*rand()/(double)RAND_MAX - 1.0;
				double randY = 2.0*rand()/(double)RAND_MAX - 1.0;
				double randZ = 2.0*rand()/(double)RAND_MAX - 1.0;
				double randfactor = 2.0*rand()/(double)RAND_MAX - 1.0;
				
				// this is a different way to do this. get a random vector on the plane and multiply by random factor of radius
				R3Vector randomVect(randX, randY, randZ);
				inPlane.Cross(randomVect);
				inPlane.Normalize();
				inPlane *= light->radius * randfactor;
				inPlane += light->position.Vector();
				
				//fprintf(stderr, "%f %f\n", (inPlane.Point()-light->position).Length(), randX);
				R3Vector areaLightFrom = intersection->position - inPlane.Point();
				double areaD = areaLightFrom.Length();
				areaLightFrom.Normalize();
				R3Vector areaLightTo = areaLightFrom;
				areaLightTo.Flip();
				
				R3Intersection *shadowI = checkShadowRay(scene, intersection->position, areaLightTo, areaD);
				if (shadowI->hit)
					continue;
				else counter += 1;
			}
			double cosine = Lfrom.Dot(direction);
			double factor = light->constant_attenuation + light->linear_attenuation * d + light->quadratic_attenuation * d * d;
			double proportion = counter/ 16.0;			
			if (cosine > 0 && factor != 0) {
				lightColor = light->color * cosine * proportion /factor;
			}
		}
		// calculate the mirror of Lto across N
		R3Vector R = Lfrom - 2*Lfrom.Dot(N) * N;
		R.Normalize();
		
		// calculate diffuse and specular contributions
		R3Rgb diffuse = m->kd * N.Dot(Lto) * lightColor;
		
		// do texture mapping if sphere has a texture
		if (intersection->node->shape->type == R3_SPHERE_SHAPE && m->texture) {
			
			R3Sphere *sph = intersection->node->shape->sphere;
			R3Vector hit = intersection->position.Vector();
			hit.Normalize();
		
			R2Image *tex = m->texture;
			double r= sph->Radius();
			double x = intersection->position.X() - sph->Center().X();
			double y = intersection->position.Y() - sph->Center().Y();
			double z = intersection->position.Z() - sph->Center().Z();
			double twoPi = 2 * M_PI;

			double v = acos(z/r)/ M_PI;
			double u = atan2(y, x)/twoPi + .5;
				
			int iU = (int)(u*tex->Width());
			int iV = (int)(v*tex->Height());
			
			diffuse *= tex->Pixel(iU, iV);
		}
		R3Rgb specular = m->ks * pow(V.Dot(R), m->shininess)* lightColor;
		// add if appropriate
		if (N.Dot(Lto) >= 0) {
			color += diffuse;
			
		}
		if (V.Dot(R) > 0)
			color += specular;
	}
	return color;
}

R3Intersection *IntersectMesh(R3Ray *ray, R3Node *node) {
	R3Intersection *firstIntersect = new R3Intersection();
	R3Mesh *mesh = node->shape->mesh;
	firstIntersect->t = DBL_MAX;
	firstIntersect->hit = false;
	// check intersection of each face
	for (int i = 0; i < mesh->NFaces(); i++) {
		R3MeshFace *face = mesh->Face(i);
		// find intersection in plane
		R3Intersection *intersect = IntersectPolygon(ray, face);
		if (intersect->hit) {
			if (intersect->t < firstIntersect->t) {
				firstIntersect = intersect;
			}
		}
	}
	firstIntersect->node = node;
	return firstIntersect;
}
// returns the t for a ray-plane intersection
double IntersectPlane(R3Ray *ray, R3Plane p) {
	R3Point P0 = ray->Start();
	double numerator = -(P0.Vector().Dot(p.Normal()) + p.D());
	double denominator = ray->Vector().Dot(p.Normal());
	if (denominator == 0)
		return -1;
	return numerator/denominator;
}
// used for mesh intersection
R3Intersection *IntersectPolygon(R3Ray *ray, R3MeshFace *face) {
	R3Intersection *intersect = new R3Intersection();
	intersect->hit = false;
	R3Point P0 = ray->Start();
	R3Plane p = face->plane;
	double t = IntersectPlane(ray, p);
	if (t < 0)
		return intersect;
	intersect->t = t;
	intersect->normal = p.Normal();
	R3Point point = P0 + t*ray->Vector().Point();
	
	// check if point is inside triangle
	for (unsigned int j = 0; j < face->vertices.size(); j++) {
		int previous = j - 1;
		if (j == 0) previous = face->vertices.size() - 1;
		R3Point T1 = face->vertices[previous]->position;
		R3Point T2 = face->vertices[j]->position;
		R3Vector V1 = T1 - P0;
		R3Vector V2 = T2 - P0;
		V2.Cross(V1);
		R3Vector N1 = V2;
		N1.Normalize();
		double sign = ray->Vector().Dot(N1);
		if (sign < 0) {
			intersect->hit = false;
			return intersect;
		}
			
	}
	intersect->hit = true;
	intersect->position = point;
	return intersect;
}
// intersect cylinder
R3Intersection *IntersectCylinder(R3Ray *ray, R3Node *node) {
	R3Intersection *i = new R3Intersection();
	i->hit = false;
	i->t = DBL_MAX;
	
	R3Cylinder *cylinder = node->shape->cylinder;
	double radius = cylinder->Radius();
	double height = cylinder->Height();
	R3Point O = cylinder->Center();
	
	// y value of the base and apex
	double base = O.Y() - height/2.0;
	double apex = O.Y() + height/2.0;
	
	R3Point P0 = ray->Start();
	R3Point Pv = ray->Vector().Point();
	double p0x = P0.X() - O.X();
	double p0z = P0.Z() - O.Z();
	
	double a = Pv.X()*Pv.X() + Pv.Z()*Pv.Z();
	double c = p0x*p0x + p0z*p0z - radius * radius;
	double b = 2*p0x*Pv.X() + 2*p0z*Pv.Z();
	double determinant = b*b - 4*a*c;
	
	R3Point topCenter(0, apex, 0);
	R3Point bottomCenter(0, base, 0);
	R3Vector normal = cylinder->Axis().Vector();

	R3Plane topCap(topCenter, normal);
	normal.Flip();
	R3Plane bottomCap(bottomCenter, normal);

	
	// check if intersection exists
	if (determinant < 0) {
		i->hit = false;
	}
	// if camera is not in cylinder
	else if (-b - sqrt(determinant) > 0) {
		i->t = (-b - sqrt(determinant))/(2*a);
		i->hit = true;
		i->position = ray->Point(i->t);
		R3Point displacement(0, i->position.Y(), 0);
		R3Vector norm = i->position - (O + displacement);
		norm.Normalize();
		i->normal = norm;
		i->node = node;
	}
	// if cylinder is not behind camera
	else if ((-b + sqrt(determinant))/(2*a) > 0) {
		i->t = (-b + sqrt(determinant))/(2*a); 
		i->hit = true;
		i->position = ray->Point(i->t);
		R3Point displacement(0, i->position.Y(), 0);
		R3Vector norm = i->position - (O + displacement);
		norm.Normalize();
		i->normal = norm;
		i->node = node;
	}
	
	// make sure intersection falls in range of finite cylinder
	if (i->position.Y() < base || i->position.Y() > apex) {
		i->hit = false;
	}
	
	double t = IntersectPlane(ray, topCap);
	if (t > 0) {

		R3Vector distance = ray->Point(t) - topCenter;

		if (distance.Length() <= radius) {
			if (i->hit && i->t > t || !i->hit) {
				i->hit = true;
				i->t = t;
				i->position = ray->Point(t);
				i->normal = topCap.Normal();
				i->node = node;
			}
		}
	}
	
	t = IntersectPlane(ray, bottomCap);
	if (t > 0) {
		R3Vector distance = ray->Point(t) - bottomCenter;
		if (distance.Length() <= radius) {
			if (i->hit && i->t > t || !i->hit) {
				i->hit = true;
				i->t = t;
				i->position = ray->Point(t);
				i->normal = bottomCap.Normal();
				i->node = node;
			}
		}
	}
	return i;
}

R3Intersection *IntersectCone(R3Ray *ray, R3Node *node) {
	R3Intersection *i = new R3Intersection();
	i->hit = false;
	i->t = DBL_MAX;
	
	R3Cone *cone = node->shape->cone;
	double radius = cone->Radius();
	double r2 = radius * radius;
	double height = cone->Height();
	double h2 = height * height;
	R3Point O = cone->Center(); 
	double slope = height/radius;
	
	// y value of the base and apex
	double base = O.Y() - height/2.0;
	double apex = O.Y() + height/2.0;
	O.SetY(base); // set O as the center of the base
	
	R3Point P0 = ray->Start();
	R3Point Pv = ray->Vector().Point();
	double p0x = P0.X() - O.X();
	double p0y = P0.Y() - O.Y() - height;
	double p0z = P0.Z() - O.Z();

	double a = Pv.X()*Pv.X() + Pv.Z()*Pv.Z() - Pv.Y()*Pv.Y()*r2/h2;
	double c = p0x*p0x + p0z*p0z - p0y*p0y*r2/h2;
	double b = 2*p0x*Pv.X() + 2*p0z*Pv.Z() - 2*p0y*Pv.Y()*r2/h2;
	double determinant = b*b - 4*a*c;
	
	R3Point topCenter(0, base, 0);
	R3Vector normal = cone->Axis().Vector();
	normal.Flip();
	R3Plane topCap(topCenter, normal);

	// check if intersection exists
	if (determinant < 0) {
		return i;
	}
	// if camera is not in cylinder
	if (-b - sqrt(determinant) > 0) {
		i->t = (-b - sqrt(determinant))/(2*a);
		i->hit = true;
		
		i->position = ray->Point(i->t);
		double yPos = i->position.Y();
		R3Point center = O;
		center.SetY(yPos - (apex - yPos)/height*radius/slope);
		R3Vector norm = i->position - center;
		norm.Normalize();
		i->normal = norm;
		i->node = node;
	}
	// if cylinder is not behind camera
	else if ((-b + sqrt(determinant))/(2*a) > 0) {
		i->t = (-b + sqrt(determinant))/(2*a); 
		i->hit = true;
		i->position = ray->Point(i->t);
		double yPos = i->position.Y();
		R3Point center = O;
		center.SetY(yPos - (apex - yPos)/height*radius/slope);
		R3Vector norm = center - i->position;
		norm.Normalize();
		i->normal = norm;
		i->node = node;
	}
	
	// make sure intersection falls in range of finite cylinder
	if (i->position.Y() < base || i->position.Y() > apex) {
		i->hit = false;
	}
	double t = IntersectPlane(ray, topCap);
	if (t > 0) {

		R3Vector distance = ray->Point(t) - O;
		if (distance.Length() < radius) {
			if (i->hit && i->t > t || !i->hit) {
				i->hit = true;
				i->t = t;
				i->position = ray->Point(t);
				i->normal = topCap.Normal();
				i->node = node;
				
			}
		}
	}
	return i;
}

R3Intersection *IntersectBox(R3Ray *ray, R3Node *node) {
	R3Intersection *i = new R3Intersection();
	i->hit = false;
	i->t = DBL_MAX;
	R3Box *box = node->shape->box;
	// check each side of the box
	R3Point bbL = box->Min(); // stands for front bottom left
	R3Point bbR = box->Min();
	bbR.SetX(box->Max().X());
	R3Point btL = box->Min();
	btL.SetY(box->Max().Y());
	R3Point btR = box->Max();
	btR.SetZ(box->Min().Z());
	R3Point fbL = box->Min();
	fbL.SetZ(box->Max().Z());
	R3Point fbR = box->Max();
	fbR.SetY(box->Min().Y());
	R3Point ftL = box->Max();
	ftL.SetX(box->Min().X());
	R3Point ftR = box->Max();
	
	R3Vector frontV(0, 0, 1);
	R3Vector backV(0, 0, -1);
	R3Vector leftV(-1, 0, 0);
	R3Vector rightV(1, 0, 0);
	R3Vector topV(0, 1, 0);
	R3Vector bottomV(0, -1, 0);
	
	R3Plane front(fbL, frontV);
	if (ray->Vector().Dot(front.Normal()) < 0) {
		double t = IntersectPlane(ray, front);
		R3Point p = ray->Point(t);
		if (p.X() > fbL.X() && p.X() < fbR.X() && t > 0) {
			if (p.Y() > fbL.Y() && p.Y() < ftL.Y()) {
				if (i->hit && t < i->t || !i->hit) {
					i->hit = true;
					i->t = t;
					i->position = p;
					i->normal = front.Normal();
				}
			}
		}
	}
	
	R3Plane back(bbL, backV);
	if (ray->Vector().Dot(back.Normal()) < 0) {
		double t = IntersectPlane(ray, back);
		R3Point p = ray->Point(t);
		if (p.X() > bbL.X() && p.X() < bbR.X() && t > 0) {
			if (p.Y() > bbL.Y() && p.Y() < btL.Y()) {
				if (i->hit && t < i->t || !i->hit) {
					i->hit = true;
					i->t = t;
					i->position = p;
					i->normal = back.Normal();
				}
			}
		}
	}
	
	R3Plane left(bbL, leftV);
	if (ray->Vector().Dot(left.Normal()) < 0) {
		double t = IntersectPlane(ray, left);
		R3Point p = ray->Point(t);
		if (p.Z() > bbL.Z() && p.Z() < fbL.Z() && t > 0) {
			if (p.Y() > bbL.Y() && p.Y() < btL.Y()) {
				if (i->hit && t < i->t || !i->hit) {
					i->hit = true;
					i->t = t;
					i->position = p;
					i->normal = left.Normal();
				}
			}
		}
	}
	R3Plane right(bbR, rightV);
	if (ray->Vector().Dot(right.Normal()) < 0) {
		double t = IntersectPlane(ray, right);
		R3Point p = ray->Point(t);
		if (p.Z() > bbR.Z() && p.Z() < fbR.Z() && t > 0) {
			if (p.Y() > bbR.Y() && p.Y() < btR.Y()) {
				if (i->hit && t < i->t || !i->hit) {
					i->hit = true;
					i->t = t;
					i->position = p;
					i->normal = right.Normal();
				}
			}
		}
	}
	R3Plane bottom(bbL, bottomV);
	if (ray->Vector().Dot(bottom.Normal()) <= 0) {
		double t = IntersectPlane(ray, bottom);
		R3Point p = ray->Point(t);
		if (p.X() > bbL.X() && p.X() < fbR.X() && t > 0) {
			if (p.Z() > bbL.Z() && p.Z() < fbL.Z()) {
				if (i->hit && t < i->t || !i->hit) {
					i->hit = true;
					i->t = t;
					i->position = p;
					i->normal = bottom.Normal();
				}
			}
		}
	}
	R3Plane top(btL, topV);
	if (ray->Vector().Dot(top.Normal()) <= 0) {
		double t = IntersectPlane(ray, top);
		R3Point p = ray->Point(t);
		if (p.X() > btL.X() && p.X() < ftR.X() && t > 0) {
			if (p.Z() > btL.Z() && p.Z() < ftL.Z()) {
				if (i->hit && t < i->t || !i->hit) {
					i->hit = true;
					i->t = t;
					i->position = p;
					i->normal = top.Normal();
				}
			}
		}
	}
	
	if (i->hit) {
		i->node = node;
	}
	return i;
		
}

R3Intersection *IntersectBoundingBox(R3Ray *ray, R3Box bbox) {
	R3Intersection *i = new R3Intersection();
	i->hit = false;
	i->t = DBL_MAX;
	R3Box box = bbox;
	// check each side of the box
	
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
	if (ray->Vector().Dot(front.Normal()) < 0) {
		double t = IntersectPlane(ray, front);
		R3Point p = ray->Point(t);
		if (p.X() > fbL.X() && p.X() < fbR.X() && t > 0) {
			if (p.Y() > fbL.Y() && p.Y() < ftL.Y()) {
				if (i->hit && t < i->t || !i->hit) {
					i->hit = true;
					i->t = t;
					i->position = p;
					i->normal = front.Normal();
				}
			}
		}
	}
	
	R3Plane back(bbL, backV);
	if (ray->Vector().Dot(back.Normal()) < 0) {
		double t = IntersectPlane(ray, back);
		R3Point p = ray->Point(t);
		if (p.X() > bbL.X() && p.X() < bbR.X() && t > 0) {
			if (p.Y() > bbL.Y() && p.Y() < btL.Y()) {
				if (i->hit && t < i->t || !i->hit) {
					i->hit = true;
					i->t = t;
					i->position = p;
					i->normal = back.Normal();
				}
			}
		}
	}
	
	R3Plane left(bbL, leftV);
	if (ray->Vector().Dot(left.Normal()) < 0) {
		double t = IntersectPlane(ray, left);
		R3Point p = ray->Point(t);
		if (p.Z() > bbL.Z() && p.Z() < fbL.Z() && t > 0) {
			if (p.Y() > bbL.Y() && p.Y() < btL.Y()) {
				if (i->hit && t < i->t || !i->hit) {
					i->hit = true;
					i->t = t;
					i->position = p;
					i->normal = left.Normal();
				}
			}
		}
	}
	R3Plane right(bbR, rightV);
	if (ray->Vector().Dot(right.Normal()) < 0) {
		double t = IntersectPlane(ray, right);
		R3Point p = ray->Point(t);
		if (p.Z() > bbR.Z() && p.Z() < fbR.Z() && t > 0) {
			if (p.Y() > bbR.Y() && p.Y() < btR.Y()) {
				if (i->hit && t < i->t || !i->hit) {
					i->hit = true;
					i->t = t;
					i->position = p;
					i->normal = right.Normal();
				}
			}
		}
	}
	R3Plane bottom(bbL, bottomV);
	if (ray->Vector().Dot(bottom.Normal()) <= 0) {
		double t = IntersectPlane(ray, bottom);
		R3Point p = ray->Point(t);
		if (p.X() > bbL.X() && p.X() < fbR.X() && t > 0) {
			if (p.Z() > bbL.Z() && p.Z() < fbL.Z()) {
				if (i->hit && t < i->t || !i->hit) {
					i->hit = true;
					i->t = t;
					i->position = p;
					i->normal = bottom.Normal();
				}
			}
		}
	}
	R3Plane top(btL, topV);
	if (ray->Vector().Dot(top.Normal()) <= 0) {
		double t = IntersectPlane(ray, top);
		R3Point p = ray->Point(t);
		if (p.X() > btL.X() && p.X() < ftR.X() && t > 0) {
			if (p.Z() > btL.Z() && p.Z() < ftL.Z()) {
				if (i->hit && t < i->t || !i->hit) {
					i->hit = true;
					i->t = t;
					i->position = p;
					i->normal = top.Normal();
				}
			}
		}
	}
	
	return i;
		
}


R3Intersection *IntersectSphere(R3Ray *ray, R3Node *node) {
	R3Sphere *sphere = node->shape->sphere;
	R3Intersection *i = new R3Intersection();
	R3Point O = sphere->Center();
	double radius = sphere->Radius();
	R3Point P0 = ray->Start();
	R3Vector L = O - P0;
	double tca = L.Dot(ray->Vector());

	// check if intersects behind the camera
	/*if (tca < 0) {
		i->hit = false;
		return i;
	}*/
	// check if thc can be found
	double dSquare = L.Dot(L) - tca*tca;
	if (dSquare > radius * radius) {
		i->hit = false;
		return i;
	}
	double thc = sqrt(radius * radius - dSquare);

	// check the smaller intersection, return if greater than 0
	if (tca - thc > 0) {
		double t = tca - thc;
		i->hit = true;
		i->node = node;
		i->position = ray->Point(t);
		i->normal = i->position - O;
		i->normal.Normalize();
		i->t = t;
		return i;
		
	}
	if (tca + thc > 0) {
		double t = tca + thc;
		i->hit = true;
		i->node = node;
		i->position = ray->Point(t);
		i->normal = O - i->position;
		i->normal.Normalize();
		i->t = t;
		return i;
	}
	i->hit = false;
	return i;
	
}


