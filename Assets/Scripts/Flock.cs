using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using static Feesh;

public class Flock : MonoBehaviour
{
	public static string JSON_PATH;

	public int seed;
	public GameObject predictPrefab;
	public bool enablePrediction;

	[Tooltip("Your general feesh flock agent.")]
	public Feesh feeshPrefab;
	[Tooltip("Only use this if you want this Flock to have a leader type. Automatically makes feesh remain within this Flock")]
	public Feesh feeshLeaderPrefab;
	List<Feesh> feeshes = new List<Feesh>();
	public FlockingBehavior behavior;
	[Range(1, 1000)]
	[Tooltip("How many agents are in the flock")]
	public int flockCount = 250;
	const float feeshDensity = 0.08f;
	[Range(1f, 100f)]
	[Tooltip("Increase the velocity of the flock agents")]
	public float velocityMultiplier = 10f;
	[Range(1f, 100f)]
	[Tooltip("The maximum velocity of the flock agents")]
	public float maxVelocity = 5f;
	[Range(1f, 10f)]
	[Tooltip("The radius agents will accept neighbors in")]
	public float neighborRadius = 1.5f;
	[Range(0f, 1f)]
	[Tooltip("The radius agents will consider other agents to be too close")]
	public float avoidanceRadiusMultiplier = 0.5f;
	[Range(0f, 100f)]
	[Tooltip("The radius agents will avoid obstacles")]
	public float obstacleAvoidanceRadius = 5f;
	[Range(0f, 100f)]
	[Tooltip("The radius the flock will attempt to stay within")]
	public float stayWithinRadius = 5f;
	[Range(0f, 100f)]
	[Tooltip("The radius of the leader the fish will follow up to")]
	public float followToRadius = 5f;
	[Tooltip("Apply movement smoothing when following a leader.")]
	public bool smoothFollow = false;
	[Tooltip("Fish agents will group up with other fish agents who have the same tag, regardless of what Flock instance they are apart of, but ignore any other fish. This applies to the tags of the fish agent type the flock spawns, not the tag of the flock itself.")]
	public bool useTags = false; //Use this if you want your feesh to only group with other feesh of the same tag (can be outside this Flock object)
	[Tooltip("Fish agents will exclusively group up with fish spawned by their Flock class object. This takes takes precedence over tags.")]
	public bool stayWithinThisFlock = false; //Use this if you want your feesh to only stay within the same instance of the Flock class (takes priority over useTags)
	[Tooltip("Fish agents will respect the above two discriminators. If disabled, these agents will continue to group up with agents who are ignoring them")]
	public bool respectDiscriminators = true; //Use this if you want agents of this flock to ignore flocks with useTags or stayWithinThisFlock set to true;
	Feesh newLeader;
	public LayerMask ObsticleLayer;
	public LayerMask FishLayer;

	float squareMaxVelocity;
	float squareNeighborRadius;
	float squareAvoidanceRadius;
	float squareObstacleAvoidanceRadius;

	public float captureRate = 1f/5f;
	private float counter = 0;

	public Feesh GetLeader
	{
		get
		{
			return newLeader;
		}
	}
	public float getSquareAvoidanceRadius
	{
		get
		{
			return squareAvoidanceRadius;
		}
	}

	public float getSquareObstacleAvoidanceRadius
	{
		get
		{
			return squareObstacleAvoidanceRadius;
		}
	}

		
	private float[][][] pTimeline;
	private int timestep_counter;

	void Start()
	{
		if (enablePrediction)
		{
			JSON_PATH = Application.dataPath + "/.." + "/SwarmData.json";

			try
			{
				using (StreamReader r = new StreamReader(JSON_PATH))
				{
					string json = r.ReadToEnd();
					pTimeline = JsonUtility.FromJson<float[][][]>(json);
				}

				if (pTimeline.Length != flockCount)
				{
					pTimeline = null;
					Debug.LogError("Loaded timeline length is not equal to current swarm :(");
				}
			}
			catch (Exception e)
			{
				pTimeline = null;
				Debug.LogError(e.Message + "\n" + e.StackTrace);
			}
		}
		else
		{
			pTimeline = null;
		}

		
		timestep_counter = 5;


		UnityEngine.Random.InitState(seed);
		squareMaxVelocity = maxVelocity * maxVelocity;
		squareNeighborRadius = neighborRadius * neighborRadius;
		squareObstacleAvoidanceRadius = obstacleAvoidanceRadius * obstacleAvoidanceRadius;
		squareAvoidanceRadius = squareNeighborRadius * avoidanceRadiusMultiplier * avoidanceRadiusMultiplier;
		if (feeshLeaderPrefab != null)
		{
			stayWithinThisFlock = true;
			newLeader = Instantiate(
				feeshLeaderPrefab, //Our feesh prefab
				UnityEngine.Random.insideUnitSphere * flockCount * feeshDensity + this.gameObject.transform.position,  //A random position in sphere (with size based on number of feesh) to spawn it at
				UnityEngine.Random.rotation, //Facing a random direction
				transform //With this class as the parent
			);
			newLeader.InitializeFlock(this);
		}
		for (int i = 0; i < flockCount; i++)
		{ //For the number of feesh we want
			Feesh newFeesh = Instantiate(
				feeshPrefab, //Our feesh prefab
				UnityEngine.Random.insideUnitSphere * flockCount * feeshDensity + this.gameObject.transform.position,  //A random position in sphere (with size based on number of feesh) to spawn it at
				UnityEngine.Random.rotation, //Facing a random direction
				transform //With this class as the parent
			);
			newFeesh.name = "feesh " + i;
			newFeesh.InitializeFlock(this); //This instance of the Flock class is the Flock this feesh belongs to
			feeshes.Add(newFeesh);
		}
	}

	void FixedUpdate()
	{
		Feesh.int_count++;
		if (Feesh.int_count > Feesh.CAP_INTERVAL)
		{
			Feesh.int_count = 0;
			if (pTimeline != null && timestep_counter < pTimeline[0].Length)
			{
				for (int i = 0; i < pTimeline.Length; i++)
				{
					Vector3 pos = new Vector3(
						pTimeline[i][timestep_counter][0],
						pTimeline[i][timestep_counter][1],
						pTimeline[i][timestep_counter][2]
					);
					Vector3 dir = PVector.GetVector3(
						pTimeline[i][timestep_counter][3],
						pTimeline[i][timestep_counter][4],
						pTimeline[i][timestep_counter][5]
					);
					Instantiate(predictPrefab, pos, Quaternion.LookRotation(dir, Vector3.up), null);
				}
			}
			timestep_counter++;
		}

		for (int i = 0; i < feeshes.Count; i++)
		{ //Oh god, a loop for for every single frame.
			Feesh feesh = feeshes[i];
			Collider[] nearbyObstacleColliders = Physics.OverlapSphere(feesh.transform.position, obstacleAvoidanceRadius, ObsticleLayer);

			List<Transform> nearby = GetNearbyObjects(feesh, nearbyObstacleColliders); //List of everything near current feesh
			Vector3 direction = behavior.CalculateDirection(feesh, nearby, this); //Calculate our direction
			direction *= velocityMultiplier;
			if (direction.sqrMagnitude > squareMaxVelocity)
			{ //If greater than max velocity
				direction = direction.normalized * maxVelocity; //Normalize (set it to 1) and set to max speed
			}
			feesh.Move(nearbyObstacleColliders, direction); //Move in direction
		}
		//if (newLeader != null)
		//{
		//	newLeader.Move(nearbyObstacleColliders, newLeader.transform.forward);
		//}
	}

	List<Transform> GetNearbyObjects(Feesh feesh, Collider[] nearbyObstacleColliders)
	{
		List<Transform> nearby = new List<Transform>(); //List of nearby transforms.
		int nearFishCount = 0;
		Collider[] nearbyNeighborColliders = Physics.OverlapSphere(feesh.transform.position, neighborRadius, FishLayer); //Array of nearby colliders around current fish within neighbor radius

		foreach (Collider collider in nearbyNeighborColliders)
		{
			if (useTags == false && stayWithinThisFlock == false)
			{ //If flock indescriminately groups up
				if (respectDiscriminators == false)
				{ //If ignores discriminators of other flocks
					nearFishCount++;
					nearby.Add(collider.transform); //Add transform of nearby collider to nearby list and follow anyways
				}
				else
				{ //If we respect flocks with discriminators
					Feesh feeshInstance = collider.GetComponent<Feesh>(); //Get Feesh object from its collider
					if (feeshInstance != null && feeshInstance.getFlock.stayWithinThisFlock == false && feeshInstance.getFlock.useTags == false)
					{ //If collider belongs to a feesh and this feesh does not use discriminators
						nearFishCount++;
						nearby.Add(collider.transform);
					}
				}
			}
			else if (stayWithinThisFlock == true)
			{ //Else if grouping up by class instance
				Feesh feeshInstance = collider.GetComponent<Feesh>();
				if (feeshInstance != null && feeshInstance.getFlock == feesh.getFlock)
				{ //If collider belongs to a feesh and is part of same class
					nearFishCount++;
					nearby.Add(collider.transform);
				}
			}
			else if (feesh.CompareTag(collider.tag))
			{ //Otherwise filter by tag
				//An agent with the same tag may belong to a different flock which wishes to stay within that class. Need to check 
				if (respectDiscriminators == false)
				{  //If we ignore discriminators
					nearFishCount++;
					nearby.Add(collider.transform);
				}
				else
				{
					Feesh feeshInstance = collider.GetComponent<Feesh>();
					if (feeshInstance != null && feeshInstance.getFlock.stayWithinThisFlock == false && feeshInstance.getFlock.useTags == false)
					{ //If collider belongs to a feesh and this feesh does not use discriminators
						nearFishCount++;
						nearby.Add(collider.transform);
					}
				}
			}
		}
		feesh.NearbyCount = nearFishCount;
		foreach (Collider collider in nearbyObstacleColliders)
		{
			if (collider.gameObject.layer.Equals(LayerMask.NameToLayer("Obstacle"))) { //If an obstalce
				nearby.Add(collider.transform); //Add collider no matter what
			}
		}

		return nearby;
	}

	private void OnDestroy()
	{
		string json = "[\n";
		
		for (int i = 0; i < feeshes.Count; i++)
		{
			json += feeshes[i].TimeToString() + ",\n";
		}
		json = json.Substring(0, json.Length - 2);
		json += "\n]";

		Debug.Log(json);

		using (System.IO.StreamWriter writetext = new System.IO.StreamWriter(JSON_PATH, append: false))
		{
			writetext.WriteLine(json);
		}
	}

}
