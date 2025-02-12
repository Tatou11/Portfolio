using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.AI;

public class IA : IA_Utils
    //Gere tout les autre script IA,TODO faire que ce script spawn tout les autre script et leur donne leur paramètre , sauf celui de l'arme a feux
{
    [Header("IA Script")]
    [SerializeField] private IA ia_Main_Script;
    [SerializeField] private IA_Vie ia_Vie;
    [SerializeField] private Gun ia_Gun;
    [SerializeField] private IA_Gestion_Equipement ia_Equipement;
    [SerializeField] private Spawn_enemy spawner;


    [Header("IA Management")]
    private GameObject joueur;
    private NavMeshAgent nav;
    [SerializeField] private Animator anim;
    [SerializeField] private GameObject enemy_Animation_Ragdoll;
    [Header("IA Healf Variable")]
    [SerializeField] private int max_Healf = 100;
    [Header("IA Equipement Variable")]
    [SerializeField] private Equipement_Type helmet_Preset = Equipement_Type.Random;
    [SerializeField] private Equipement_Type vest_Preset = Equipement_Type.Random;
    [SerializeField] private Equipement_Type visor_Preset = Equipement_Type.Random;
    [SerializeField] private Equipement_Type weapon_Preset = Equipement_Type.Random;

    [Header("Equippement and weapon placement")]
    [SerializeField] private GameObject helmet_Empty;
    [SerializeField] private GameObject visor_Empty;
    [SerializeField] private GameObject vest_Empty;
    [SerializeField] private GameObject weapon_Empty;

 

    [Header("IA Deplacement Variable")]
    [SerializeField] private float speed = 3.5f;
    [SerializeField] private float angular_Speed = 120;
    [SerializeField] private float acceleration = 8;
    [SerializeField] private float stop_Distance = 5;
    [SerializeField] private float distance_Detection = 20;

    private float distance = 100;
    private float timer = 0;

    public Gun Ia_Gun { get => ia_Gun; set => ia_Gun = value; }
    public GameObject Joueur { get => joueur; set => joueur = value; }


    // Start is called before the first frame update
    void Start()
    {
        joueur = GameObject.FindGameObjectWithTag("Player");
        nav = gameObject.GetComponent<NavMeshAgent>();
        ia_Main_Script = gameObject.GetComponent<IA>();
        ia_Vie = gameObject.GetComponent<IA_Vie>();
        ia_Equipement = gameObject.GetComponent<IA_Gestion_Equipement>();
        anim = enemy_Animation_Ragdoll.GetComponent<Animator>();
        
        //
        ia_Vie.Set_Up(ia_Main_Script, max_Healf);
        ia_Equipement.Set_Up(helmet_Preset, visor_Preset, vest_Preset, weapon_Preset, helmet_Empty, visor_Empty, vest_Empty, weapon_Empty);
        //Deplacement
        nav.speed = speed;
        nav.angularSpeed = angular_Speed;
        nav.acceleration = acceleration;
        nav.stoppingDistance = stop_Distance;
        Ragdoll_Activator(false);
        
        //Desactivate ragdoll

    }

    // Update is called once per frame
    void FixedUpdate()
    {
        if (ia_Gun == null)//If in start it is too fast
        {
            ia_Gun = weapon_Empty.GetComponentInChildren<Gun>();//too soon
            anim.SetBool(ia_Gun.GetComponent<Gun>().Gun_Scriptable.gun_Type, true);
            enemy_Animation_Ragdoll.AddComponent<EnemyIK>();//Because we don't have the gun wet
            enemy_Animation_Ragdoll.GetComponent<EnemyIK>().Set_Up(anim, Ia_Gun.Right_Hand, Ia_Gun.Left_Hand);           
        }

        //print(nav.velocity.magnitude);
        if (nav.velocity.magnitude <= 0)
        {
            anim.SetBool("Idle", true);
            anim.SetBool("Walking", false);
        }
        else
        {            
            anim.SetBool("Idle", false);
            anim.SetBool("Walking", true);
        }


        
        nav.updateRotation = true;
        timer = timer + Time.fixedDeltaTime;
        distance = Vector3.Distance(gameObject.transform.position, joueur.transform.position);

        if (distance < distance_Detection)
        {
            nav.SetDestination(joueur.transform.position);
        }
        else
        {
            if (timer > 5)
            {
                Vector3 point_Aleatoire = Random.insideUnitSphere * 10;
                point_Aleatoire += gameObject.transform.position;
                nav.SetDestination(point_Aleatoire);
                timer = 0;
            }
        }
    }
    private void Ragdoll_Activator(bool activ)
    {
        Collider[] colliders = enemy_Animation_Ragdoll.GetComponentsInChildren<Collider>();
        foreach (Collider col in colliders)
        {
            //col.enabled = activ;
        }
        Rigidbody[] rigidbodies = enemy_Animation_Ragdoll.GetComponentsInChildren<Rigidbody>();
        foreach (Rigidbody rb in rigidbodies)
        {
            rb.isKinematic = !activ; // Rend kinematic si désactivé
        }

    }
    public void Set_Up(Spawn_enemy sp, Equipement_Type helmet_, Equipement_Type vest_, Equipement_Type visor_, Equipement_Type weapon_)
    {
        spawner = sp;
        //TODO a faire plus tard quand plus d'arme
        helmet_Preset = helmet_;
        vest_Preset = vest_;
        visor_Preset = visor_;
        weapon_Preset = weapon_;
}
    public void Die()
    {
        weapon_Empty.SetActive(false);
        //Destroy(weapon_Empty);
        gameObject.AddComponent<Rigidbody>();
        nav.enabled = false;
        if(spawner != null)
        {
            spawner.Enemy_Die(gameObject);//Prévien le spawner de la mort
        }
        anim.enabled = false;     
        ia_Vie.enabled = false;
        ia_Equipement.enabled = false;
        ia_Gun.Drop_Gun(false,true);
        gameObject.AddComponent<Kill>();//TODO a changer plus tard, pour laisser les cadavre
        gameObject.GetComponent<Kill>().Timer_Max = 10;
        ia_Gun.enabled = false;
        Ragdoll_Activator(true);


        ia_Main_Script.enabled = false;
        //Drop the gun        
    }
}