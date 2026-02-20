import cv2
import mediapipe as mp
import time
import os

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

CHEMIN_MODELE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")


dernieres_mains = []

def sur_resultat(result, output_image, timestamp_ms):
    global dernieres_mains
    if result.hand_landmarks:
        donnees_mains = []
        for pts_main in result.hand_landmarks:
            pos_x = pts_main[0].x
            donnees_mains.append((pos_x, pts_main))
        dernieres_mains = donnees_mains
    else:
        dernieres_mains = []

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=CHEMIN_MODELE),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=sur_resultat,
)

camera = cv2.VideoCapture(0)

# les etats du jeu
ETAT_ATTENTE = 0
ETAT_COMPTE_REBOURS = 1
ETAT_CAPTURE = 2
ETAT_RESULTAT = 3

etat = ETAT_ATTENTE
debut_compte = 0
debut_capture = 0
debut_resultat = 0

choix_j1 = None
choix_j2 = None
texte_gagnant = ""
score_j1 = 0
score_j2 = 0

DUREE_COMPTE = 3
DUREE_CAPTURE = 0.7
DUREE_RESULTAT = 2.5

IDS_BOUTS = [4, 8, 12, 16, 20]


def compter_doigts(points_main):
    doigts = []

    # pouce
    if points_main[IDS_BOUTS[0]].x < points_main[IDS_BOUTS[0] - 1].x:
        doigts.append(1)
    else:
        doigts.append(0)

    # les autres doigts
    for i in range(1, 5):
        if points_main[IDS_BOUTS[i]].y < points_main[IDS_BOUTS[i] - 2].y:
            doigts.append(1)
        else:
            doigts.append(0)

    return sum(doigts)


def dessiner_main(img, points_main, largeur, hauteur, couleur):
    connexions = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (5,9),(9,10),(10,11),(11,12),
        (9,13),(13,14),(14,15),(15,16),
        (13,17),(17,18),(18,19),(19,20),
        (0,17),
    ]
    points = [(int(lm.x * largeur), int(lm.y * hauteur)) for lm in points_main]

    for conn in connexions:
        cv2.line(img, points[conn[0]], points[conn[1]], couleur, 2)
    for point in points:
        cv2.circle(img, point, 5, couleur, -1)


def obtenir_choix(nb_doigts):
    if nb_doigts <= 1:
        return "Pierre"
    elif nb_doigts == 2:
        return "Ciseaux"
    else:
        return "Papier"


def trouver_gagnant(j1, j2):
    if j1 == j2:
        return "Egalite!"
    if (j1 == "Pierre" and j2 == "Ciseaux") or \
       (j1 == "Ciseaux" and j2 == "Papier") or \
       (j1 == "Papier" and j2 == "Pierre"):
        return "Joueur 1 Gagne!"
    return "Joueur 2 Gagne!"


COULEUR_J1 = (255, 100, 0)
COULEUR_J2 = (0, 100, 255)

timestamp_image = 0

with HandLandmarker.create_from_options(options) as detecteur:
    while True:
        ok, image = camera.read()
        if not ok:
            break

        image = cv2.flip(image, 1)
        hauteur, largeur, _ = image.shape
        milieu_x = largeur // 2
        maintenant = time.time()

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        timestamp_image += 1
        detecteur.detect_async(image_mp, timestamp_image)

        cv2.line(image, (milieu_x, 0), (milieu_x, hauteur), (100, 100, 100), 1)

        cv2.putText(image, "J1", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COULEUR_J1, 2)
        cv2.putText(image, "J2", (largeur - 50, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COULEUR_J2, 2)

        main_j1 = None
        main_j2 = None
        for pos_x, pts in dernieres_mains:
            if pos_x < 0.5:
                main_j1 = pts
            else:
                main_j2 = pts

        if main_j1 is not None:
            dessiner_main(image, main_j1, largeur, hauteur, COULEUR_J1)
        if main_j2 is not None:
            dessiner_main(image, main_j2, largeur, hauteur, COULEUR_J2)

        if etat == ETAT_ATTENTE:
            cv2.putText(image, "Appuyer ESPACE pour jouer", (milieu_x - 200, hauteur // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image, "Chaque joueur d'un cote", (milieu_x - 210, hauteur // 2 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        elif etat == ETAT_COMPTE_REBOURS:
            temps_ecoule = maintenant - debut_compte
            restant = DUREE_COMPTE - temps_ecoule
            if restant > 0:
                chiffre = str(int(restant) + 1)
                cv2.putText(image, chiffre, (milieu_x - 40, hauteur // 2 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 8)
            else:
                etat = ETAT_CAPTURE
                debut_capture = maintenant
                choix_j1 = None
                choix_j2 = None

        elif etat == ETAT_CAPTURE:
            temps_ecoule = maintenant - debut_capture
            cv2.putText(image, "MONTREZ!", (milieu_x - 70, hauteur // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

            if main_j1 is not None:
                choix_j1 = obtenir_choix(compter_doigts(main_j1))
            if main_j2 is not None:
                choix_j2 = obtenir_choix(compter_doigts(main_j2))

            if temps_ecoule >= DUREE_CAPTURE:
                if choix_j1 is None:
                    choix_j1 = "?"
                if choix_j2 is None:
                    choix_j2 = "?"
                if choix_j1 != "?" and choix_j2 != "?":
                    texte_gagnant = trouver_gagnant(choix_j1, choix_j2)
                    if texte_gagnant == "Joueur 1 Gagne!":
                        score_j1 += 1
                    elif texte_gagnant == "Joueur 2 Gagne!":
                        score_j2 += 1
                else:
                    texte_gagnant = "Main pas detectee!"
                etat = ETAT_RESULTAT
                debut_resultat = maintenant

        elif etat == ETAT_RESULTAT:
            temps_ecoule = maintenant - debut_resultat

            if texte_gagnant == "Joueur 1 Gagne!":
                couleur_gagnant = COULEUR_J1
            elif texte_gagnant == "Joueur 2 Gagne!":
                couleur_gagnant = COULEUR_J2
            elif texte_gagnant == "Egalite!":
                couleur_gagnant = (0, 255, 255)
            else:
                couleur_gagnant = (0, 0, 255)

            cv2.putText(image, f"J1: {choix_j1}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, COULEUR_J1, 2)
            cv2.putText(image, f"J2: {choix_j2}", (largeur - 250, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, COULEUR_J2, 2)
            cv2.putText(image, texte_gagnant, (milieu_x - 180, hauteur // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, couleur_gagnant, 3)

            if temps_ecoule >= DUREE_RESULTAT:
                etat = ETAT_ATTENTE

        cv2.putText(image, f"J1: {score_j1}  -  J2: {score_j2}", (milieu_x - 100, hauteur - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(image, "Q=Quitter", (largeur - 90, hauteur - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("Pierre Papier Ciseaux - 2 Joueurs", image)

        touche = cv2.waitKey(1) & 0xFF
        if touche == ord('q'):
            break
        elif touche == ord(' ') and etat == ETAT_ATTENTE:
            etat = ETAT_COMPTE_REBOURS
            debut_compte = time.time()

camera.release()
cv2.destroyAllWindows()
