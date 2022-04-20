using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
[RequireComponent(typeof(Collider))] //Needed to see what's around the feesh
public class Feesh : MonoBehaviour
{
	public const int OBSTACLE_COUNT = 4;

	public static int int_count = 0;


	Flock thisFlock; //The Flock object/class this feesh belongs to
	Collider feeshCollider;
	int nearbyCount = 0;

	private List<float[]> timesteps;

	public Collider getFeeshCollider {
		get {
			return feeshCollider;
		}
	}
	public Flock getFlock {
		get {
			return thisFlock;
		}
	}
	public void InitializeFlock(Flock flock) {
		thisFlock = flock;
	}
	// Start is called before the first frame update
	void Start()
	{
		feeshCollider = GetComponent<Collider>();
		timesteps = new List<float[]>();
	}

	public int NearbyCount { 
		get {
			return nearbyCount;
			}
		set {
			nearbyCount = value;
		}
	}

	public void Move(Collider[] nearbyObstacleColliders, Vector3 velocity)
	{
		transform.forward = velocity;
		transform.position += velocity * Time.fixedDeltaTime; //Add our velocity to our current position (framerate independent)

		if (int_count == thisFlock.cap_interval)
			timesteps.Add(GetTimestepValues(nearbyObstacleColliders, velocity));
	}

	public float[] GetTimestepValues(Collider[] nearbyObstacleColliders, Vector3 velocity)
	{
		if (thisFlock.obstaclesAsNodes)
		{
			return Get7PointTimestep(transform.position, velocity, feeshCollider.bounds.extents.magnitude);
		}

		PVector[] result = new PVector[OBSTACLE_COUNT + 2];

		//result[0] = new PVector(transform.position);
		result[0] = PVector.AsVector3(transform.position);
		result[1] = new PVector(velocity);

		Array.Copy(GetObsticles(nearbyObstacleColliders), 0, result, 2, OBSTACLE_COUNT);

		float[] timestep = new float[result.Length * 3];

		for (int i = 0; i < result.Length; i++)
		{
			timestep[i * 3 + 0] = result[i].theta;
			timestep[i * 3 + 1] = result[i].phi;
			timestep[i * 3 + 2] = result[i].r;
		}

		return timestep;
	}

	public string TimeToString()
	{
		return TimeToString(timesteps);
	}

	public static string TimeToString(List<float[]> time)
	{
		string result = "\t[\n";
		foreach (float[] step in time)
		{
			result += "\t\t[";
			foreach (float f in step)
				result += string.Format("{0}, ", f);
			result = result.Substring(0, result.Length - 2);
			result += "],\n";
		}

		result = result.Substring(0, result.Length - 2);
		result += "\n\t]";

		return result;
	}

	public PVector[] GetObsticles(Collider[] nearbyObstacleColliders, int count = OBSTACLE_COUNT)
	{
		Array.Sort(nearbyObstacleColliders,
			(x, y) => (int)((Vector3.Distance(transform.position, x.bounds.center) - Vector3.Distance(transform.position, y.bounds.center)) * 100)
		);

		PVector[] result = new PVector[count];
		for (int i = 0; i < count; i++)
		{
			if (i >= nearbyObstacleColliders.Length)
			{
				result[i] = new PVector();
				continue;
			}
			else
			{
				Vector3 diff = nearbyObstacleColliders[i].ClosestPoint(transform.position) - transform.position;
				result[i] = new PVector(diff);
			}
		}

		return result;
	}


	public static float[] Get7PointTimestep(Vector3 pos, Vector3 vel, float radius)
	{
		PVector polarVel = new PVector(vel);

		return new float[7] {
			pos.x,
			pos.y,
			pos.z,
			polarVel.theta,
			polarVel.phi,
			polarVel.r,
			radius
		};
	}

	[Serializable]
	public struct PVector
	{
		public PVector(Vector3 vector3)
		{
			r = vector3.magnitude;
			theta = Mathf.Acos(vector3.z / r);
			phi = Mathf.Atan2(vector3.y, vector3.x);

			if (theta == float.NaN) theta = 0.0f;
			if (phi == float.NaN) phi = 0.0f;
		}

		public float theta;
		public float phi;
		public float r;

		public Vector3 GetVector3()
		{
			return new Vector3(
				r * Mathf.Cos(phi) * Mathf.Sin(theta),
				r * Mathf.Sin(phi) * Mathf.Sin(theta),
				r * Mathf.Cos(theta)
			);
		}

		public static Vector3 GetVector3(float theta, float phi, float r)
		{
			return new Vector3(
				r * Mathf.Cos(phi) * Mathf.Sin(theta),
				r * Mathf.Sin(phi) * Mathf.Sin(theta),
				r * Mathf.Cos(theta)
			);
		}

		public static PVector AsVector3(Vector3 vector)
		{
			return new PVector()
			{
				theta = vector.x,
				phi = vector.y,
				r = vector.z
			};
		}
	}
}
