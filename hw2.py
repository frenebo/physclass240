import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize

def linear_t_funcs(v_meas):
    return np.array([
        np.power(v_meas,-2.5),
        np.power(v_meas,-2.5)*np.log10(v_meas),
        np.power(v_meas,-2.5)*np.square(np.log10(v_meas)),
        np.power(v_meas,-4.5),
        np.power(v_meas,-2),
    ])

def linear_analysis(freqs_measured,temps_measured):
    A = linear_t_funcs(freqs_measured).T
    U,Wvals,VT = np.linalg.svd(A)
    Winv = 1/Wvals
    threshold = 1e-14
    Winv[Wvals < threshold*np.amax(Wvals)] = 0
    Winv_full = np.zeros(A.T.shape,dtype=Winv.dtype)
    Winv_full[:len(Wvals),:len(Wvals)] = np.diag(Winv)
    abest = np.linalg.multi_dot([VT.T,Winv_full,U.T,temps_measured])
    print("Best fit for a coefficients:")
    print(abest)
    predicted_T =  linear_t_funcs(freqs_measured).T@abest
    residual = temps_measured - predicted_T

    res_rms = np.sqrt(np.mean(np.power(residual,2)))

    fig, ax = plt.subplots()
    ax.set_xlabel(r'Frequency, $\nu$ (MHz)')
    ax.set_ylabel(r'Temperature, $T$ (K)')
    ax.set_ylim([-0.3,0.3])
    ax.plot(freqs_measured,residual,label="r.m.s. = {0:.2g}".format(res_rms))
    ax.legend()

def nonlinear_pred_T(v_meas,b0,b1,b2,b3,b4):
    term1 = np.power(v_meas, (-2.5 +b1+b2*np.log10(v_meas)) )
    term2 = np.exp(-b3*np.power(v_meas,-2))
    term3 = np.power(v_meas,-2)
    return b0*term1*term2 + b3*term3

def nonlinear_analysis(freqs_measured,temps_measured):
    print(freqs_measured.min())
    popt,pcov = scipy.optimize.curve_fit(nonlinear_pred_T, freqs_measured,temps_measured,method='lm',maxfev=50000)
    b0best,b1best,b2best,b3best,b4best = popt
    predicted_T = np.array([nonlinear_pred_T(v,b0best,b1best,b2best,b3best,b4best) for v in freqs_measured])

    residual = temps_measured - predicted_T

    res_rms = np.sqrt(np.mean(np.power(residual,2)))

    fig, ax = plt.subplots()
    ax.set_xlabel(r'Frequency, $\nu$ (MHz)')
    ax.set_ylabel(r'Temperature, $T$ (K)')
    ax.set_ylim([-0.4,0.4])
    ax.plot(freqs_measured,residual,label="r.m.s. = {0:.2g}".format(res_rms))
    ax.legend()

def main():
    plt.style.use("phys240.mplstyle")
    with open("skytemperature.dat") as f:
        freqs_measured = []
        temps_measured = []
        for line in list(f)[1:]:
            if len(line.rstrip()) == 0:
                continue

            F,T = line.rstrip().split()
            freqs_measured.append(float(F))
            temps_measured.append(float(T))
        freqs_measured = np.array(freqs_measured)
        temps_measured = np.array(temps_measured)

    linear_analysis(freqs_measured,temps_measured)
    nonlinear_analysis(freqs_measured,temps_measured)
    plt.show()


if __name__ == "__main__":
    main()