#include "PhysicsNode.h"
#define DAMP_FACTOR 0.9995

PhysicsNode::PhysicsNode(void)	{
	target = NULL;
	init_DefaultValues();

}

PhysicsNode::PhysicsNode(Quaternion orientation, Vector3 position) {
	m_orientation	= orientation;
	m_position		= position;
	init_DefaultValues();


	yaw -= (Window::GetMouse()->GetRelativePosition().x);
	pitch -= (Window::GetMouse()->GetRelativePosition().y);
	pitch = min(pitch,90.0f);
	pitch = max(pitch,-90.0f);

	if(yaw < 0) {
		yaw += 360.0f;
	}
	if(yaw > 360.0f) {
		yaw -= 360.0f;
	}
}

PhysicsNode::~PhysicsNode(void)	{

	delete s1;

}

void	PhysicsNode::init_DefaultValues() {
	
	m_colli_radius = 1.0f;
	s1 = new CollisionSphere(m_position,m_colli_radius);
	collBox = new CollisionAABB(m_position,Vector3(m_colli_radius));
	hit = false;

}


void	PhysicsNode::Update(float msec) {


		if(target) {
			target->SetTransform(BuildTransform());
		}
}

Matrix4		PhysicsNode::BuildTransform() {
	Matrix4 m = m_orientation.ToMatrix();

	m.SetPositionVector(m_position);

	return m;
}