import math as m
import numpy as np
import matplotlib.pyplot as plt
import cv2
#from scipy.signal import savgol_filter
import scipy.optimize as opt

grad_cut = 2

def cut(x): #Schneidet alle Nachkommastellen nach der grad_cut -ten ab, fuer die Anzeige der Fitparameter
     return int(x*10**grad_cut)/10**grad_cut

def color_rec(hsv,farbe): # Funktion, welche eine nach Farbe definierte Stelle ermittelt.
     hue_min, hue_max, sat_min, sat_max, val_min, val_max = farbe # Die als Tupel uebergebene Farbe wird entpackt.
     lower = np.array([hue_min, sat_min, val_min]) # Untere schranke wird festgelegt.
     upper = np.array([hue_max, sat_max, val_max]) # Obere schranke wird festgelegt.
     mask =  cv2.inRange(hsv, lower, upper)   	                        # Erstellt maske, welche für Werte innerhalb des Intervalls# den wert 1 annimmt, sonst 0.

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


video_name = 'MAH04184_Trim1.mp4'
ymin = 520 #Der für das Video 'MAH04184_Trim1.mp4' zu betrachtende Ausschnitt des Bildes
ymax = 600 
xmin = 0
xmax = 1500

dist = 34.7 #Entfernung zwischen Kamera und Startvorrichtung
alpha = 1.0/220.0*np.arctan(4.216/dist) #Winkel pro Pixel

Dichte_N2 = 807 #Dichte von fluessigem Stickstoff in kg/m^3
Dichte_W  = 1000 #Dichte von Wasser in kg/m^3
Dichte_Luft = 1.25

V_W = 650e-6 #Volumen des eingefüllten Wassers in m^3
V_N2 = 150e-6 #Volumen des fluessigen Stickstoffs in m^3
g = 9.81 #Ortsfaktor in m/s^2

t0s = 0.0 #Falls das Video nicht so zugeschnitten ist, dass die Rakete bei t = 0 startet
h0s = -12.25  #Starthöhe der Rakete


masse_treibstoff = V_N2*Dichte_N2+V_W*Dichte_W #Masse des Treibstoffes in kg
me = 0.297-0.067 #Leermasse der Rakete in kg
m0 = me+masse_treibstoff #Startmasse in kg


def hs1(x, a): #Funktion, anstatt if-Bedingung, sodass die Funktion noch gefittet werden kann
     #ist 1, wenn x > a und 0, wenn x < a
     return (np.sign(x-a)+1.0)/2.0 #
     #return 0.5 + np.arctan(100000000*(x-a))/np.pi #Auch so möglich

def hs2(x, a):
     #ist 1, wenn x < a und 0, wenn x > a
     return (np.sign(a-x)+1.0)/2.0
     #return 0.5 + np.arctan(100000000*(a-x))/np.pi


r = [] #Array speichert die Höhe der Rakete über die Zeit
T = 11.0/359 #Zeit in s zwischen zwei Frames
xAx = [] #Zeit zu der ein Höhenwert gespeichert wird
count = 0 #Zählt die Einträge
rote_farbe = (150,180,50,125,200,250)  #)  #HSV Parameter der Rakete in 'MAH04177_Trim.mp4', über die Maske in der color_rec Methode bestimmbar	
cap = cv2.VideoCapture(video_name) # Videodatei wird geoeffnet.
cv2.namedWindow('Tracking',cv2.WINDOW_NORMAL) # Benennt ein Fenster
cv2.resizeWindow('Tracking', 600,600) # Gibt dem Fenster mit Namen 'Tracking' die Abmessungen in Pixel. (Hier wird nur die Anzeige geandert.)
while(True):
     ret, frame = cap.read()
     if frame is None:  # Wenn kein Bild mehr da ist gibt cap.read() ein None zurück und hier wird die Schleife abgebrochen.
          break			
     frame2 = frame[ymin:ymax, xmin:xmax] #frame[ymin:ymax, xmin:xmax]
     hsv = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV) # Das Bild wird zur besseren Verarbeitung in das hsv-Format konvertiert.
     #rote_position.append(color_rec(hsv,rote_farbe))
     i,j = color_rec(hsv,rote_farbe)
     if(i != 0):
          k1 = i-720 #Differenz in Pixeln zur Mitte des Bildes
          x1 = np.tan(k1*alpha) #Bestimmt die Höhe der Rakete aus dem Pixel, an dem sie lokalisiert wurde
          r.append(dist*x1 - h0s) #Höhe der Rakete wird im Array gespeichert
          xAx.append(T*count) #Zeitpunkt für diese Höhenmessung wird gespeichert
     zeichne_kreis(frame2,(i,j))   # An die Position, welche im letzten Listenelement der jeweiligen Farbe steht, wird ein Kreis gezeichnet.
     cv2.imshow('Tracking',frame2) # Ergebnis wird angezeigt.
     count += 1
     if cv2.waitKey(1) & 0xFF == ord('q'): #'q' zum Beenden drücken
          break
#Ignorieren des letzten Wertes:
xAx = xAx[:-1]
r = r[:-1]
plt.plot(xAx, r, '.', label='Messpunkte')
#plt.plot(xAx, r)


#Höhenkurve-----------------------------------------------------
h0s = 0 #Die Kurve wird so angezeigt, dass die Rakete bei h = 0 m startet

def h_tot(t, T, ve): #Ohne Fallunterscheidung definierte Höhe der Rakete zum Zeitpunkt t, bei der Brenndauer T und der Ausstoßgeschwindigkeit ve
     x = t-t0s
     q = masse_treibstoff/T

     hf = ve*(m0/q-T)*np.log(1-q*T/m0)+ ve*T - 0.5*g*T**2 + h0s #Hoehe der Rakete bei Abschluss der Brennphase
     vf = ve*np.log(m0/me) - g*T #Geschwindigkeit der Rakete bei Abschluss der Brennphase
     ret = hs2(x, 0)*h0s + hs1(x, 0)*hs2(x, T)*(ve*(m0/q-x)*np.log(abs(1-q*x/m0))+ ve*x - 0.5*g*x**2 + h0s) + hs1(x, T)*(hf + vf*(x-T) - 0.5*g*(x-T)**2)
     return ret


#------------------------------------------------------------------------------------

p, cm = opt.curve_fit(h_tot, xAx, r) #Fit der Funktion
ve = p[1] #Fitparameter
T = p[0]
Y = h_tot(np.array(xAx), T, ve) #Array enthaelt die y-Werte der Fitfunktion


plt.plot(xAx, Y, label='Fitfunktion ergibt $T$ = '+str(cut(T))+' s und $v_e$ = '+str(cut(ve))+' m/s')

plt.xlabel('$t$ in s')
plt.ylabel('$h$ in m')


cap.release() # Der Zugang zur Videodatei wird beendet.
cv2.destroyAllWindows() # Alle Fenster werden geschlossen.

plt.legend()
plt.show()

