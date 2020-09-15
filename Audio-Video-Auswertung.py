import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.optimize as opt
from scipy.io import wavfile
import scipy.fftpack

#=============Audio==================================

#Die Aufnahme wird z.B. mit dem Programm Audacity gemacht und als .wav abgespeichert. In Audacity kann man auch schon die Zeit ablesen, in der der Flug ist und sich ein Spektrogramm ueber die Zeit anzeigen lassen.

filename = '2-13-02.wav' #Die Aufnahme enthaelt u.A. die Reflexion am Brett und den Fall danach
a = int(35.4*44100) #Der verwertbare Teil der Aufnahme beginnt bei 35.4 Sekunden #1561000 #
b = int(38.1*44100) #Der verwertbare Teil der Aufnahme endet bei 35.4 Sekunden #1690000
c = 600 #Wir betrachten das Frequenzintervall von der 600.-ten Frequenz zur 900.-ten
d = 900

f_S = 3380.0 #Frequenz mit der der Piezolautsprecher sendet

stereo = 1 #1 fuer Stereoaufnahmen, 0 fuer Monoaufnahmen
sos = 346.0 # = 20.05*(24.7+273.15)**0.5 #Schallgeschwindigkeit in m/s

N_gone_through_it = 10000 #Wie viele 'Frames' werden in jeder Iteration betrachtet
N_dist = 1000 #Um so viele 'Frames' sind die betrachteten Intervalle auseinander

def doppler(f): #Berechnet die Geschwindigkeit, wenn die Frequenz f gemessen wird
    return sos*(1.0-f_S/f)

#fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1) #Erstellt 4 Plots: Geschwindigkeit, Frequenz, Fouriertransformation und Hoehe, nur relevant bei der Einstellung der Parameter

def calc():
    fs, data1 = wavfile.read(filename) #fs ist die Sample-Frequenz, hier immer 44100 Hz
    T = 1.0/fs 
    fmax = fs/2.0 #Nach dem Nyquist-Shannon-Abtasttheorem kann maximal die Frequenz fs/2 untersucht werden
    if(stereo == 1): #Wenn es sich um eine Stereo-Aufnahme handelt muessen die beiden Spuren in ein Array ueberlagert werden
        data0 = SteMo(data1)
    else:
        data0 = data1
    velocities = [] #Array enthaelt die Geschwindigkeit der Rakete
    frequencies = [] #Array enthaelt die maximalen Frequenzen im untersuchten Frequenzintervall, also die gemessene Frequenz des Piezolautsprechers
    time = []
    Length = int((b-a)/N_dist) #Anzahl an Iterationen durch die Aufnahme
    for i in range(Length): #Iteration durch den relevanten Bereich
        start = a+i*N_dist #Intervall beginnt im 'start'-ten Frame
        #print(start/fs)
        data = data0[start:start+N_gone_through_it] #Betrachtete Daten
        x = np.linspace(0.0, N_gone_through_it*T, N_gone_through_it)
        y0 = np.abs(scipy.fftpack.fft(data))
        xf = np.linspace(0.0, fmax, N_gone_through_it//2)[c:d] #Array mit untersuchten Frequenzen
        yf = y0[:len(y0)//2][c:d] #Vorfaktoren zu den Frequenzen

        f_max = xf[np.max(np.where(yf == np.amax(yf)))] #Bei dieser Frequenz ist der Vorfaktor maximal 
        #print(f_max)
        frequencies.append(f_max)
        velocities.append(doppler(f_max))
        time.append(start/fs)
        #plt.plot(xf, yf[:len(yf)//2])
        """
        #Diser Abschnitt sollte zum einstellen der Parameter nicht auskommentiert sein
        ax1.plot(time, velocities) #plottet die Geschwindikeit ueber die Zeit, im oberen Graph 
        ax2.plot(time, frequencies) #plottet die Frequenz ueber die Zeit, im mittleren Graph (wichtig um zu ueberpruefen, ob der richtige Teil der Aufnahme gewaehlt wurde)
        ax3.plot(xf, yf) #plottet die Fouriertransformation des Intervalls (wichtig um zu ueberpruefen, ob der richtige Frequenzbereich gewaehlt wurde)
        plt.draw()
        plt.pause(0.000001)
        ax1.cla() #loescht die Plots wieder, damit sie ersetzt werden koennen
        ax2.cla()
        ax3.cla()
        """
    return [time, velocities, frequencies]


def SteMo(A):#Ueberlagert ein Array mit Stereo Aufnahmen in eins mit Mono Aufnahmen
    x = []
    for i in range(0, len(A)):
        x.append(A[i][0]+A[i][1])
    return x 

A = calc()


H = []
H.append(0)

for i in range(len(A[0])-1): #Numerische Integration der Geschwindigkeit ergibt die Hoehe
    H.append(H[-1] + A[1][i]*(A[0][i+1]-A[0][i])) #Da A[0] das Array mit den Zeiten ist, ist dt = A[0][i+1]-A[0][i]

#===========

X_Audio = np.array(A[0]) -35.4 + 0.45 #Verschiebungsterme noetig um Audioauswertung und Videoauswertung zu ueberlagern
Y_Audio = np.array(H) + 10.0

plt.plot(X_Audio, Y_Audio, '--', label='Audioauswertung')


#ax1.plot(A[0], A[1]) #Relevant zur Analyse (v-t, f-t, h-t) am Ende irrelevant
#ax2.plot(A[0], A[2])
#ax4.plot(A[0], H)


#=======Video=======================================

def color_rec(hsv,farbe): # Funktion, welche eine nach Farbe definierte Stelle ermittelt.
     hue_min, hue_max, sat_min, sat_max, val_min, val_max = farbe # Die als Tupel uebergebene Farbe wird entpackt.
     lower = np.array([hue_min, sat_min, val_min]) # Untere schranke wird festgelegt.
     upper = np.array([hue_max, sat_max, val_max]) # Obere schranke wird festgelegt.
     mask =  cv2.inRange(hsv, lower, upper)   	                        # Erstellt maske, welche fuer Werte innerhalb des Intervalls# den wert 1 annimmt, sonst 0.

     #cv2.namedWindow('Farberkennung',cv2.WINDOW_NORMAL) 					# Benennt ein Fenster.
     #cv2.resizeWindow('Farberkennung', 600,600) 							# Gibt dem Fenster mit Namen 'Farberkennung' die Abmessungen in Pixel.
     #cv2.imshow('Farberkennung',mask)	

     y_werte, x_werte = np.where(mask == 255)							# Es werden die x- und y-Werte ausgelesen, welche ein True (255) bekomen haben.
     if len(x_werte) != 0 and len(y_werte) != 0:
          y_mittel = int(np.mean(y_werte))								# Es wird der Mittelwert aus allen y-Werten gebildet.
          x_mittel = int(np.mean(x_werte))								# Es wird der Mittelwert aus allen x-Werten gebildet.
          position = (x_mittel, y_mittel)									# Die mittlere Position aller Trues entspricht dem Tupel beider Mittelwerte.
     else:
          position = (0,0)												# Wenn kein Wert gefunden, wird hier (0,0) als Position gewaehlt.		
     return position														# Ergebnis wird zurueckgegeben.

def zeichne_kreis(frame, position): # Funktion, welche einen Kreis zeichnet.
     cv2.circle(frame, position, 25, (0,0,255), 4)   # Wer mag kann Radius, Farbe und Dicke anpassen (letzten drei Argumente). Farbe hier als RGB kodiert.



video_name = 'MAH04177_Trim1.mp4'
ymin = 550 #Der fuer das Video 'MAH04177_Trim.mp4' zu betrachtende Ausschnitt des Bildes
ymax = 570 
xmin = 100
xmax = 1350

h0s = -11.0 #Starthoehe der Rakete

dist = 34.7 #Entfernung zwischen Kamera und Startvorrichtung
alpha = 1.0/220.0*np.arctan(4.216/dist) #Winkel pro Pixel

r = [] #Array speichert die Hoehe der Rakete ueber die Zeit
T = 11.0/359 #Zeit in s zwischen zwei Frames
xAx = [] #Zeit zu der ein Hoehenwert gespeichert wird
count = 0 #Zaehlt die Eintraege
rote_farbe = (150,180,50,125,200,250)  #HSV Parameter der Rakete in 'MAH04177_Trim.mp4', ueber die Maske in der color_rec Methode bestimmbar
cap = cv2.VideoCapture(video_name) # Videodatei wird geoeffnet.
cv2.namedWindow('Tracking',cv2.WINDOW_NORMAL) # Benennt ein Fenster
cv2.resizeWindow('Tracking', 600,600) # Gibt dem Fenster mit Namen 'Tracking' die Abmessungen in Pixel. (Hier wird nur die Anzeige geandert.)
while(True):
     ret, frame = cap.read()
     if frame is None:  # Wenn kein Bild mehr da ist gibt cap.read() ein None zurueck und hier wird die Schleife abgebrochen.
          break			
     frame2 = frame[ymin:ymax, xmin:xmax] #frame[ymin:ymax, xmin:xmax]
     hsv = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV) # Das Bild wird zur besseren Verarbeitung in das hsv-Format konvertiert.
     #rote_position.append(color_rec(hsv,rote_farbe))
     i,j = color_rec(hsv,rote_farbe)
     if(i != 0): #Durch optische Stoereffekte, kann die Rakete nicht immer lokalisiert werden, wenn sie nicht lokalisiert wird, setzt das Programm ihre Position auf (0,0), diese Positionen sollen nicht betrachtet werden
          k1 = i-720 #Differenz in Pixeln zur Mitte des Bildes
          x1 = np.tan(k1*alpha) #Bestimmt die Hoehe der Rakete aus dem Pixel, an dem sie lokalisiert wurde
          r.append(dist*x1 - h0s) #Hoehe der Rakete wird im Array gespeichert
          xAx.append(T*count) #Zeitpunkt fuer diese Hoehenmessung wird gespeichert
     zeichne_kreis(frame2,(i,j))   # An die Position, welche im letzten Listenelement der jeweiligen Farbe steht, wird ein Kreis gezeichnet.
     cv2.imshow('Tracking',frame2) # Ergebnis wird angezeigt.
     count += 1
     if cv2.waitKey(1) & 0xFF == ord('q'): #'q' zum Beenden druecken
          break

        
#Ignorieren des letzten Wertes:
xAx = np.array(xAx[:-1])-0.428 #Verschiebung zum Vergleich mit der Audioauswertung
r = np.array(r[:-1])
plt.plot(xAx, r, '--', label='Videoauswertung')

plt.xlabel('$t$ in s')
plt.ylabel('$h$ in m')
plt.legend() 
plt.show()














