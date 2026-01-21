# Valgfag-Vision-Project
Kort guide til at køre vision-programmet (Coca-Cola / Pepsi).

---

## Sådan kører du programmet

###1) Start programmet
Kør dette script fra projektmappen:

```bash
python run_images.py

2) Vælg mode ved start
Når programmet starter, bliver du bedt om at vælge mode:
1 = Normal mode
Viser kun det færdige resultat: polygon + label på billedet
2 = Debug mode
Viser først debug-visning (feature matches) for de fundne detections
Viser derefter normal result-visning

3) Navigation (tastatur)
Når et vindue er åbent (både i normal og debug):
Højre pil / D / L → næste billede
Venstre pil / A / J → forrige billede
Q eller ESC → afslut program (eller hop ud af debug-visning)
Tip: Hvis piletaster ikke virker på din maskine, brug A/D eller J/L.

4) Hvad du ser på skærmen
Normal mode (resultat)
Programmet tegner polygon omkring detektionen
Programmet skriver en label med brandnavn (fx “Coca-Cola” / “Pepsi”)
Hvis der ikke findes noget: vises “NO DETECTIONS”
Debug mode (fejlsøgning)
Der vises et debug-vindue pr. detection, hvor du kan se feature matching
Formålet er at se, om matches er relevante (og om der er mange/få gode matches)
Du kan stadig navigere frem/tilbage med samme taster

5) Afslut
Tryk Q eller ESC i et vindue for at afslutte.