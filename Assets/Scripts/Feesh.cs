using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
[RequireComponent(typeof(Collider))] //Needed to see what's around the feesh
public class Feesh : MonoBehaviour
{
	public const int OBSTACLE_COUNT = 0;

	public static int int_count = 0;


	Flock thisFlock; //The Flock object/class this feesh belongs to
	Collider feeshCollider;
	int nearbyCount = 0;

	public List<float[]> timesteps;

	public Collider getFeeshCollider
	{
		get
		{
			return feeshCollider;
		}
	}
	public Flock getFlock
	{
		get
		{
			return thisFlock;
		}
	}
	public void InitializeFlock(Flock flock)
	{
		thisFlock = flock;
	}

	public void Awake()
	{
		timesteps = new List<float[]>();
	}

	// Start is called before the first frame update
	void Start()
	{
		feeshCollider = GetComponent<Collider>();
	}

	public int NearbyCount
	{
		get
		{
			return nearbyCount;
		}
		set
		{
			nearbyCount = value;
		}
	}

	public void Move(Collider[] nearbyObstacleColliders, Vector3 velocity)
	{
		transform.forward = velocity;
		transform.position += velocity * Time.fixedDeltaTime; //Add our velocity to our current position (framerate independent)

		if (int_count == thisFlock.cap_interval)
			thisFlock.AddTimestep(ref timesteps, GetTimestepValues(nearbyObstacleColliders, velocity));
	}

	public float[] GetTimestepValues(Collider[] nearbyObstacleColliders, Vector3 velocity)
	{
		if (thisFlock.obstaclesAsNodes)
		{
			return Get7PointTimestep(transform.position, velocity, feeshCollider.bounds.extents.magnitude);
		}

		Vector3 pos = transform.position;
		PVector polarVel = new PVector(velocity);


		float[] result = new float[(OBSTACLE_COUNT + 2) * 3];
		result[0] = pos.x;
		result[1] = pos.y;
		result[2] = pos.z;
		result[3] = polarVel.theta;
		result[4] = polarVel.phi;
		result[5] = polarVel.r;

		PVector[] pvecs = GetObsticles(nearbyObstacleColliders);
		for (int i = 0; i < pvecs.Length; i++)
		{
			result[i * 3 + 6] = pvecs[i].theta;
			result[i * 3 + 7] = pvecs[i].phi;
			result[i * 3 + 8] = pvecs[i].r;
		}

		return result;
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

	public float[][] TimeToMatrix()
	{
		return TimeToMatrix(timesteps);
	}

	public static float[][] TimeToMatrix(List<float[]> time)
	{

		return time.ToArray();
	}

	public PVector[] GetObsticles(Collider[] nearbyObstacleColliders, int count = OBSTACLE_COUNT)
	{
		PVector[] result = new PVector[count];
		if (nearbyObstacleColliders == null)
		{
			for (int i = 0; i < result.Length; i++)
			{
				result[i] = new PVector();
			}
		}
		else
		{
			Array.Sort(nearbyObstacleColliders,
				(x, y) => (int)((Vector3.Distance(transform.position, x.bounds.center) - Vector3.Distance(transform.position, y.bounds.center)) * 100)
			);
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

			if (float.IsNaN(theta)) theta = 0.0f;
			if (float.IsNaN(phi)) phi = 0.0f;
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
