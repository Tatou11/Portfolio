using System;
using System.Collections;
using UnityEngine;

public class Gun : Weapon_Utils
{
    //This script is use for both the player Gun and the IA gun, so its big but mutch easier to modifie latter
    //Maybe modify it latter for the number of "empty" variable
    [Header("Player or IA")]
    [SerializeField] private bool player = true;
    [SerializeField] private bool has_Set_Up = false;


    [Header("Weapon Variable")]
    [SerializeField] private int magazine_capacity;
     private float rate_Of_Fire;
     private int damage ;
     private float deviation;
     private AudioClip gun_sound;
     private AudioClip empty_Gun_Sound;
     private GameObject muzzle_Flash;
     private GameObject player_bullet_Trail;
     private GameObject bullet_Impact;
     private GameObject gun_Dummy_Prefab;

    [Header("Guns Object Variable")]
    [SerializeField] private Gun_Scriptable gun_Scriptable;
    private AudioSource audio_Source;
    private GameObject player_Camera;


    [Header("variable")]
    [SerializeField] private float input_Fire_Gun;
    [SerializeField] private float input_Throw_Gun;
    [SerializeField] private bool is_Selected = true;
    [SerializeField] private GameObject true_Gun;
    [SerializeField] private float timer;

    [SerializeField] private bool weapon_Picked_Up = false;
    [SerializeField] private float timer_Weapon_Pickup ;



    [Header("Specifique IA Variable")]
    [SerializeField] private GameObject player_GameObject;
    [SerializeField] private GameObject bullet_Prefab = null;
    [SerializeField] private float bullet_speed ;
    [SerializeField] private float reaload_Time ;
    [SerializeField] private Transform right_Hand;//The one wo shoot
    [SerializeField] private Transform left_Hand;//



    public bool Is_Selected { get => is_Selected; set => is_Selected = value; }
    public GameObject Player_Camera { get => player_Camera; set => player_Camera = value; }
    public Gun_Scriptable Gun_Scriptable { get => gun_Scriptable; }
    public Transform Left_Hand { get => left_Hand; set => left_Hand = value; }
    public Transform Right_Hand { get => right_Hand; set => right_Hand = value; }


    //TODO change the script so we can remove the gun when not used
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {

    }
    // Update is called once per frame
    void FixedUpdate()
    {
        if (!has_Set_Up)
        {
            Set_Up(false);
            return;
        }
        if (timer < rate_Of_Fire + 1)
        {
            timer = timer + Time.fixedDeltaTime;
        }
        if (player)//For player
        {           
            input_Fire_Gun = Input.GetAxis("Fire1");
            input_Throw_Gun = Input.GetAxis("Throw_Gun");

            
            if ((input_Fire_Gun > 0.5f) && Is_Selected && timer > rate_Of_Fire)
            {
                if (magazine_capacity == 0)
                {
                    //Play empty soundempty_Gun_Sound
                    audio_Source.PlayOneShot(empty_Gun_Sound);
                }
                else
                {
                    //Gun_Fire_Player();
                    gameObject.GetComponent<Procedural_Recoil>().ApplyRecoil();
                    Gun_Fire(player_bullet_Trail, player_Camera, true_Gun, bullet_Impact, Deviation_Computation(player_Camera), damage);
                    magazine_capacity--;
                    Gun_Firing_Effect();
                }
                timer = 0;
            }
            //Pickup
            if (weapon_Picked_Up)
            {
                timer_Weapon_Pickup = timer_Weapon_Pickup + Time.fixedDeltaTime;

            }
            if (input_Throw_Gun > 0.5f && timer_Weapon_Pickup > 0.5f)
            {
                Drop_Gun(true, false);
                timer_Weapon_Pickup = 0;
                weapon_Picked_Up = false;
            }
        }
        else//For IA
        {   //Angle for the gun
            GameObject rootTransform = gameObject.transform.root.gameObject;
            Vector3 directionToPlayer = player_GameObject.transform.position - rootTransform.transform.position;
            directionToPlayer.Normalize(); // Normaliser la direction
            float angle = Vector3.Angle(rootTransform.transform.forward, directionToPlayer);
            if (angle <= 45)
            {
                transform.parent.LookAt(player_GameObject.transform);
            }
            //directionToPlayer.Normalize();
            //TODO fix the rotation latter
            if (timer > rate_Of_Fire)
            {
                Ray ray = new Ray(true_Gun.transform.position, true_Gun.transform.forward);
                RaycastHit hit;
                
                if (Physics.Raycast(ray, out hit, 1000)&& hit.collider.tag == "Player")
                {
                    //Try to Shoot, if empty Reload 
                    if (magazine_capacity > 0)
                    {
                        //Shoot
                        Gun_Fire_IA();
                        timer = 0;
                    }
                    else
                    {
                        //Reload
                        Gun_Reload_IA();
                    }                   
                }
            }
        }       
    }
    private Vector3 Deviation_Computation(GameObject source)
    {
        Vector3 directionWithDeviation = source.transform.forward;
        directionWithDeviation.x += UnityEngine.Random.Range(-deviation, deviation);
        directionWithDeviation.y += UnityEngine.Random.Range(-deviation, deviation);
        directionWithDeviation.z += UnityEngine.Random.Range(-deviation, deviation);
        directionWithDeviation = directionWithDeviation.normalized;
        return directionWithDeviation;
    }
    private void Gun_Fire_IA()
    {
        magazine_capacity--;
        Gun_Firing_Effect(); //TEST
        GameObject projectile = Instantiate(bullet_Prefab, true_Gun.transform.position, true_Gun.transform.rotation);
        projectile.GetComponent<Rigidbody>().linearVelocity = bullet_speed * Deviation_Computation(true_Gun);
        projectile.GetComponent<Bullet>().Set_Up(gun_Scriptable);        
    }
    private void Gun_Reload_IA()
    {
        magazine_capacity = gun_Scriptable.magazine_Capacity;
        timer = -reaload_Time;//Hack
    }
    private void Gun_Firing_Effect()
    {
        audio_Source.PlayOneShot(gun_sound);
        GameObject muzzleFlash = Instantiate(muzzle_Flash, true_Gun.transform.position, true_Gun.transform.rotation);
        muzzleFlash.transform.parent = true_Gun.transform;
    }

    public void Drop_Gun(bool Throw_, bool pickable_)
    {
        GameObject weapon_Dummy_Temps = Instantiate(gun_Dummy_Prefab, true_Gun.transform.position, true_Gun.transform.rotation);//todo bug;aussi peut cumuler  les arme rammasée et c'est marrant
        weapon_Dummy_Temps.GetComponent<Weapon_Dummy>().Set_Up(Throw_, pickable_);
        if (Throw_ == true)
        {
            weapon_Dummy_Temps.GetComponent<Rigidbody>().linearVelocity = 20 * true_Gun.transform.forward;
        }
        //Desactivate player weapon
        magazine_capacity = gun_Scriptable.magazine_Capacity;
        timer = 0;
        //gameObject.GetComponent<Gun>().enabled = false;
        Destroy(gameObject);
    }
    
    
    public void Set_Up(bool is_player)
    {
        player = is_player;
        if (player)
        {
            gameObject.AddComponent<Procedural_Recoil>();
            gameObject.GetComponent<Procedural_Recoil>().Setp_Up(gun_Scriptable.recoil_Amount,gun_Scriptable.recoil_Rotation_Amount,gun_Scriptable.recoil_Speed,gun_Scriptable.shake_Amount,gun_Scriptable.shake_Duration);
            player_Camera = gameObject.transform.root.GetComponent<Main_Reference_Player>().Get_Camera();
            weapon_Picked_Up = true;
        }
        else
        {
            player_GameObject = gameObject.transform.root.GetComponent<IA>().Joueur;//TEST
            bullet_Prefab = gun_Scriptable.bullet_Prefab;
            bullet_speed = gun_Scriptable.bullet_speed;
            reaload_Time = gun_Scriptable.reaload_Time;
        }
        magazine_capacity = gun_Scriptable.magazine_Capacity;
        rate_Of_Fire = gun_Scriptable.rate_Of_Fire;
        damage = gun_Scriptable.damage;
        gun_sound = gun_Scriptable.gun_Sound;
        deviation = gun_Scriptable.deviation;
        empty_Gun_Sound = gun_Scriptable.empty_Gun_Sound;
        muzzle_Flash = gun_Scriptable.muzzle_Flash;
        player_bullet_Trail = gun_Scriptable.player_Bullet_Trail;
        bullet_Impact = gun_Scriptable.bullet_Impact;
        gun_Dummy_Prefab = gun_Scriptable.gun_Dummy_Prefab;
        audio_Source = gameObject.GetComponent<AudioSource>();
        has_Set_Up = true;
    }
}
