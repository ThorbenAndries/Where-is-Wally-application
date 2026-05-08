# Hidden Mascot Detector

Een AI-project voor school waarin een computer-vision model een zelfgemaakte mascotte zoekt in drukke afbeeldingen. De gebruiker uploadt een afbeelding, het model detecteert de mascotte intern, en de gebruiker klikt op de afbeelding om te raden waar de mascotte zit.

Het project is begonnen als een Where Is Wally/Waldo idee, maar is aangepast naar een eigen mascotte. Daardoor is de dataset beter controleerbaar, is er geen afhankelijkheid van zeldzame of auteursrechtelijk gevoelige Wally-afbeeldingen, en blijft de AI-opdracht duidelijk: zelf data maken, zelf trainen, en een interactieve toepassing bouwen.

**Waarom dit project?**
- **Eigen dataset:** de trainingsbeelden worden synthetisch gegenereerd met perfecte labels.
- **Echte AI-pipeline:** dataset maken, YOLO trainen, model gebruiken in een webapp.
- **Interactief:** de gebruiker krijgt feedback per klik: `Gevonden`, `Dichtbij` of `Ver weg`.
- **Uitbreidbaar:** later kan de synthetische data gemengd worden met eigen CVAT-labels.

**MVP**
- Genereer een dataset met een verborgen mascotte.
- Train een YOLOv8 object-detection model op de klasse `mascot`.
- Upload een afbeelding in de webapp.
- Detecteer de mascotte met het getrainde model.
- Gebruik de bounding box alleen intern.
- Laat de gebruiker klikken en geef tekst-feedback.

**Hoe het werkt**
1. `tools/generate_synthetic_mascot_dataset.py` maakt drukke zoekplaat-achtige afbeeldingen.
2. Bij elke afbeelding wordt automatisch een YOLO-label geschreven rond de mascotte.
3. YOLOv8 traint op deze dataset.
4. De FastAPI-backend laadt het model via `MODEL_PATH`.
5. De frontend toont alleen de afbeelding en stuurt klik-coördinaten naar de backend.
6. De backend vergelijkt de klik met het midden van de gedetecteerde bounding box.

**Projectstructuur**
- `app/` — FastAPI-backend voor detectie en klik-feedback.
- `frontend/` — eenvoudige HTML/JS webinterface.
- `tools/` — scripts om datasets voor te bereiden of te genereren.
- `models/` — opgeslagen modelgewichten, niet in git.
- `data/` — datasets, niet in git.
- `runs/` — YOLO-trainingsoutput, niet in git.

## Installatie

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Synthetische dataset maken

Maak een dataset met automatisch gelabelde afbeeldingen:

```bash
python3 tools/generate_synthetic_mascot_dataset.py --output data/synthetic-mascot --train 240 --val 60 --difficulty hard
```

De generator ondersteunt `easy`, `medium` en `hard`. Voor het project is `hard` het interessantst, omdat de afbeeldingen meer drukte, lookalikes en gedeeltelijke overlap bevatten.

Je kunt ook je **eigen mascotte** gebruiken. Maak daarvoor een afbeelding met transparante achtergrond, bijvoorbeeld:

```text
assets/mascot.png
```

Gebruik die mascotte dan zo:

```bash
python3 tools/generate_synthetic_mascot_dataset.py --output data/my-mascot --train 500 --val 100 --difficulty hard --mascot assets/mascot.png
```

Als je eigen achtergrondfoto's hebt, zet die bijvoorbeeld in:

```text
assets/backgrounds/
```

Dan kun je de mascotte automatisch op jouw achtergronden laten plaatsen:

```bash
python3 tools/generate_synthetic_mascot_dataset.py --output data/my-mascot --train 500 --val 100 --difficulty hard --mascot assets/mascot.png --backgrounds assets/backgrounds
```

Dat is de handigste manier om zelf data te maken zonder alles handmatig in CVAT te labelen. Het script plakt de mascotte op de afbeelding en schrijft automatisch de juiste bounding box.

De dataset krijgt deze YOLO/Ultralytics-structuur:

```text
data/synthetic-mascot/
  data.yaml
  images/train/
  images/val/
  labels/train/
  labels/val/
```

## Model trainen

Train YOLOv8 op de gegenereerde dataset:

```bash
yolo detect train data=data/synthetic-mascot/data.yaml model=yolov8n.pt epochs=30 imgsz=640
```

Na training staat het beste model meestal hier:

```text
runs/detect/train/weights/best.pt
```

Gebruik het getrainde model in de app:

```bash
MODEL_PATH=runs/detect/train/weights/best.pt TARGET_CLASS=mascot uvicorn app.main:app --reload
```

Open daarna `frontend/index.html` in de browser en upload een testafbeelding uit bijvoorbeeld:

```text
data/synthetic-mascot/images/val/
```

## Eigen CVAT-labels gebruiken

Als je later echte afbeeldingen of zelfgemaakte zoekplaten wil labelen in CVAT, kan dat nog steeds. Maak in CVAT een label met de naam `mascot`, exporteer als YOLO, pak de zip uit en zet die om:

```bash
python3 tools/prepare_cvat_yolo.py data/cvat-export --output data/cvat-mascot --fallback-class mascot
```

Daarna kun je op die dataset trainen:

```bash
yolo detect train data=data/cvat-mascot/data.yaml model=yolov8n.pt epochs=50 imgsz=640
```

Je kunt ook synthetische data gebruiken als snelle start en daarna verbeteren met CVAT-data. Dat is een sterk verhaal voor het project: eerst gecontroleerde data genereren, daarna testen of het model ook werkt op realistischer beeldmateriaal.

## Feedbacklogica

De backend berekent de afstand tussen de klik en het midden van de gedetecteerde box:

- `Gevonden`: klik zit dicht bij het midden van de box.
- `Dichtbij`: klik zit in de buurt.
- `Ver weg`: klik zit duidelijk naast de mascotte.
- `Geen detectie`: het model vond geen mascotte.
