# Spec — 15ᵉ articulation « mouth » & tâche « mouth-throw »

**Date** : 2026 (session in progress)
**Repo** : microduck_rl
**Task id** : `Mjlab-MouthThrow-Flat-MicroDuck`

## Objectif

Apprendre une compétence **one-shot** où le robot **ouvre le bec, ramasse un stylo
avec le bec, se redresse, et le lance vers le haut** avec la bouche + la tête/le cou,
le corps entier pendant qu'il raidit / se redresse. C'est un premièr pas pour rendre
le **joint `mouth` (ID 34 réel)** entraînable par RL — aujourd'hui le sim le porte
comme non-actué.

## État du sim avant ce travail (ne pas casser)

- Le modèle distribue la bouche comme **non-actuée** : `head_roll` porte toute
  l'assemblée `jaw_soft` (bec/shell/Pi/lentille), et le modèle **n'a pas de joint
  `mouth`** hé du tout. `policies/` et `duck-control` (runtime) connaissent bien la
  bouche (slot 9, 61-D obs, 14 actions), mais le sim ne la modélise pas.
- Invariant AGENTS.md : obs **61D** & action **14** partagées (hot-swap runtime).
  Les changer casserait le contrat. → Ce travail est **scopé à une seule env** qui
  utilise un **nouveau robot cfg** ; les autres envs gardent le 14-joints / 61-D.

## Décision (choisie par l'utilisateur)

Ajouter un **vrai 15ᵉ joint `mouth` actif** et le piloter. Accepte de changer la
dimension obs/action pour CETTE tâche uniquement (passage 61→~63 obs, 14→15 actions)
plutôt que de simuler la bouche par une force.

## Modèle — dérivé par `add_mouth.py` (convention post-traitement du repo)

- `src/mjlab_microduck/robot/microduck/add_mouth.py` — post-processeur texte sur le
  modèle ground-contact (même convention que `add_backlash.py`). Il sort le bec
  (`jaw`, `soft_mouth_top`, `jaw_soft`) + le site `mouth_tip` de `jaw_soft` dans un
  nouveau `<body name="mouth">`, relié par une charnière `mouth` (axe `0 1 0`,
  range « -0.52..0.52 ») + un actionneur `mouth` position (BAM).
- Résultat : `robot_groundcontact_mouth.xml`. **Validé en MuJoCo** :
  - compile avec 16 joints (freejoint + 14 + `mouth`), 15 actionneurs ;
  - piloter `mouth` déplace le bec / `mouth_tip` (0.024 m pout 0.4 rad) ;
  - la tête (`head_camera`, sur `jaw_soft`) ne bouge PAS → le bec seul suit la bouche.

## Constants

- `src/mjlab_microduck/robot/microduck_constants.py` :
  - `MICRODUCK_MOUTH_XML` + assert ;
  - `get_mouth_spec()` ;
  - `HOME_FRAME` += `r".*mouth.*": 0.0` (bec fermé au reset ; no-op pour les autres
    modèles sans joint `mouth`) ;
  - `MICRODUCK_MOUTH_ROBOT_CFG` (BAM, regex `^(?!passive_).*`) → **action 15-D**.

## Env & rewards (implémenté)

`src/mjlab_microduck/tasks/microduck_mouth_throw_env_cfg.py` — moule `ground_pick`,
commande de phase cyclique **one-shot** (`GroundPickPhaseCommandCfg(randomize_phase=False)`
→ chaque épisode commence à la phase 0), 5 segments :
`[0,LOWER_END) lower` · `[LOWER_END,GRAB_END) grab` · `[GRAB_END,WINDUP_END) rise/windup`
· `[WINDUP_END,THROW_END) throw` · `[THROW_END,1) return`.

Rewards (nouveaux mdp dans `tasks/mdp.py`, section "Mouth Throw") :
- `mouth_open_during_throw` (poids +3) — bec ouvert en lower + return ;
- `mouth_closed_during_hold` (poids +3) — bec fermé (grip) en grab..windup ;
- `mouth_throw_upward_velocity` (poids +4) — vitesse verticale du `mouth_tip` au throw ;
- `mouth_throw_peak_upward` (poids +1) — vitesse montante nette au throw (potential) ;
- `mouth_throw_ground_proximity` (poids +3) — bec près du sol, gaté sur `[0,GRAB_END)` ;
- retour debout réutilisé (`ground_pick_return_pose_*`/`_upright_phased`) gaté au return ;
- hook payload `apply_mouth_payload_release` (poids 0) — poids du stylo tenu, ON de grab,
  OFF (relâché) au throw ; masse du stylo tirée par `sample_mouth_payload` (10–40 g) ;
- régularisation `action_rate_l2`, `neck_action_rate_l2`, `joint_torques_l2`,
  `self_collisions`, `head_impact_penalty` ; stabilité (upright, body_ang_vel …).

Réutilise : `GroundPickPhaseCommand`/`_gp_phase`, `ground_pick_return_pose_*`/`_upright_phased`,
`apply_mouth_payload_force` (variante `apply_mouth_payload_release`), `sample_mouth_payload`,
`phase_pose_blend`, `_servo_joint_*`. La vitesse du `mouth_tip` est calculée par différence
finie de `site_pos_w` (l'API mjlab ne lit pas de vitesse de site) avec cache ré-armé au reset
(`reset_mouth_tip_vel_cache`).

## Non-objectifs

- Pas de grêpe par contact physique du stylo (fragile) ; le stylo = poids tenu.
- Pas de runtime Rust / `duck-control` modifié dans ce round (contrat 61-D/14 reste
  la norme de déploiement). Ce travail prouve le sim ; le déploiement est un round
  suivant (changer le runtime à 15).
- Pas de côté configurable / rough dans v1 (sol plat uniquement, comme ground_pick).

## Validation requise

- `python -m py_compile` sur tous les fichiers touchés.
- Smke réelle (machine uv+GPU) : `uv run train Mjlab-MouthThrow-Flat-MicroDuck
  --env.scene.num-envs 64 --agent.max_iterations 5` — obligatoire avant entraînement
  long, ne peut pas tourner sur ce poste.
- Tests cfg `tests/test_mouth_throw_cfg.py` (CPU, modele : `test_ground_pick_cfg.py`).

## Risques / à régler à l'entraînement

- L'ajout du joint `mouth` façon constante du HOME change la masse/tête (`mouth`
  body 0.012 kg) — vérifier la stabilité debout de la posture HOME.
- Le drop de payload au `throw_end` doit être progressif sinon le bec « lâche » trop
  fort (impulsion parasite). Timings/période à régler au réel.
- Changer obs/action casse la hot-swap runtime — bien documenter que c'est un
  contrat expérimental scoré à cette env.
