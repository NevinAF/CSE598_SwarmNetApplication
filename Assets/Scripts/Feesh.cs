using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
[RequireComponent(typeof(Collider))] //Needed to see what's around the feesh
public class Feesh : MonoBehaviour
{
	public const int OBSTACLE_COUNT = 4;
	public const int CAP_INTERVAL = 12;

	public static int int_count = 0;


	Flock thisFlock; //The Flock object/class this feesh belongs to
	Collider feeshCollider;
	int nearbyCount = 0;

	private List<PVector[]> timesteps;

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
		timesteps = new List<PVector[]>();
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

		if (int_count == CAP_INTERVAL)
			timesteps.Add(GetTimestepValues(nearbyObstacleColliders, velocity));
	}

	public PVector[] GetTimestepValues(Collider[] nearbyObstacleColliders, Vector3 velocity)
	{
		PVector[] result = new PVector[OBSTACLE_COUNT + 2];

		//result[0] = new PVector(transform.position);
		result[0] = new PVector()
		{
			theta = transform.position.x,
			phi = transform.position.y,
			r = transform.position.z
		};
		result[1] = new PVector(velocity);

		Array.Copy(GetObsticles(nearbyObstacleColliders), 0, result, 2, OBSTACLE_COUNT);

		return result;
	}

	public string TimeToString()
	{
		string result = "\t[\n";
		foreach (PVector[] step in timesteps)
		{
			result += "\t\t[";
			foreach(PVector pVector in step)
				result += string.Format("{0}, {1}, {2}, ", pVector.theta, pVector.phi, pVector.r);
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

	[Serializable]
	public struct PVector
	{
		public PVector(Vector3 vector3)
		{
			r = vector3.magnitude;
			theta = Mathf.Acos(vector3.z / r);
			phi = Mathf.Atan2(vector3.y, vector3.x);
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

		public static  Vector3 GetVector3(float theta, float phi, float r)
		{
			return new Vector3(
				r * Mathf.Cos(phi) * Mathf.Sin(theta),
				r * Mathf.Sin(phi) * Mathf.Sin(theta),
				r * Mathf.Cos(theta)
			);
		}
	}
}
