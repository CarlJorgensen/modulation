#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


def constellation(modulation_type, M):
    if modulation_type == 'PAM':
        return np.linspace(-M + 1, M - 1, M)
    elif modulation_type == 'PSK':
        return np.exp(1j * 2 * np.pi * np.arange(M) / M)
    elif modulation_type == 'QAM':
        levels = int(np.sqrt(M))
        real_levels = np.linspace(-levels + 1, levels - 1, levels)
        const = np.array([r + 1j * i for r in real_levels for i in real_levels])
        return const
    else:
        raise ValueError(f"Unsupported modulation type: {modulation_type}")


def AWGN(symbols, SNR_dB):
    signal_power = np.mean(np.abs(symbols)**2)
    SNR_linear = 10**(SNR_dB / 10)
    noise_power = signal_power / SNR_linear

    if np.iscomplexobj(symbols):
        noise_real = np.random.normal(0, np.sqrt(noise_power/2), symbols.shape[0])
        noise_imag = np.random.normal(0, np.sqrt(noise_power/2), symbols.shape[0])
        noise = noise_real + 1j * noise_imag
    else:
        noise = np.random.normal(0, np.sqrt(noise_power), symbols.shape[0])

    return symbols + noise


def optimum_detector(rx_symbols, list):
    detected_idx = np.zeros(rx_symbols.shape, dtype=int)
    detected_symbols = np.zeros(rx_symbols.shape, dtype=complex)
    for i, r in enumerate(rx_symbols):
        distances = np.abs(list - r)
        detected_idx[i] = np.argmin(distances)

        detected_symbols[i] = list[detected_idx[i]]

    return detected_idx, detected_symbols


def SER(symbols, detected_symbols):
    errors = np.sum(symbols != detected_symbols)
    ser = errors / len(symbols)
    return ser


def BER(symbols, detected_symbols, gray_codes, M):
    detected_bits = []
    sent_bits = []
    reverse_gray = {v: k for k, v in gray_codes.items()}

    detected_bits = np.vectorize(reverse_gray.get)(detected_symbols)
    sent_bits = np.vectorize(reverse_gray.get)(symbols)

    errors = np.sum(detected_bits != sent_bits)
    ber = errors / (len(symbols) * np.log2(M))
    return ber


def hamming_encode(data):
    if len(data) % 4 != 0:
        raise ValueError("Input data length should be multiple of 4.")

    G = [[1, 0, 0, 1, 0, 1, 1],
         [0, 1, 0, 1, 0, 1, 0],
         [0, 0, 1, 1, 0, 0, 1],
         [0, 0, 0, 0, 1, 1, 1]]

    encode_data = []
    for k in range(0, len(data), 4):
        block = data[k:k+4]
        for i in range(7):
            sum = 0
            for j in range(4):
                sum += block[j] * G[j][i]
            encode_data.append(sum % 2)

    return encode_data


def hamming_decode(received_data):
    if len(received_data) % 7 != 0:
        raise ValueError("Recieved data length should be multiple of 7.")

    H = [[1, 0, 1, 0, 1, 0, 1],
         [1, 1, 0, 0, 1, 1, 0],
         [1, 1, 1, 1, 0, 0, 0]]

    decoded_data = []
    for k in range(0, len(received_data), 7):
        block = received_data[k:k+7]
        syndrome = []
        for i in range(3):
            syndrome_sum = 0
            for j in range(7):
                syndrome_sum += block[j] * H[i][j]
            syndrome.append(syndrome_sum % 2)
        error_pos = sum([2 ** i for i, bit in enumerate(syndrome) if bit])
        if error_pos > 0:
            block[7-error_pos] ^= 1
        decoded_data.extend([block[0], block[1], block[2], block[4]])

    return decoded_data


def symbols_to_bits(detected_symbols, gray_codes):
    reverse_gray = {v: k for k, v in gray_codes.items()}
    binary_strings = [reverse_gray[symbol] for symbol in detected_symbols]
    return [int(bit) for binary_string in binary_strings for bit in binary_string]


def plot_ber(SNR_db, ber_encoded, ber_plain):
    min_ber = 1e-12  # Small value to represent zero BER
    ber_encoded = np.maximum(ber_encoded, min_ber)
    ber_plain = np.maximum(ber_plain, min_ber)

    plt.figure()
    plt.semilogy(SNR_db, ber_encoded, marker='o', color='red', label='Encoded')
    plt.semilogy(SNR_db, ber_plain, marker='x', color='blue', label='Plain')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate')
    plt.title('Bit Error Rate vs SNR')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_constellation(constellation1, constellation2, constellation3, constellation4,
                       annote1, annote2, annote3, annote4):
    plt.figure(figsize=(12, 12))

    plt.subplot(2, 2, 1)
    plt.scatter(np.real(constellation1), np.imag(constellation1), color='red')
    for i, txt in enumerate(annote1):
        plt.annotate(txt, (np.real(constellation1[i]), np.imag(constellation1[i]) + 0.005))
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.title('8-PAM')
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.scatter(np.real(constellation2), np.imag(constellation2), color='red')
    for i, txt in enumerate(annote2):
        plt.annotate(txt, (np.real(constellation2[i]), np.imag(constellation2[i]) + 0.05))
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.title('QPSK')
    plt.grid()

    plt.subplot(2, 2, 3)
    plt.scatter(np.real(constellation3), np.imag(constellation3), color='red')
    for i, txt in enumerate(annote3):
        plt.annotate(txt, (np.real(constellation3[i]), np.imag(constellation3[i]) + 0.05))
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.title('8-PSK')
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.scatter(np.real(constellation4), np.imag(constellation4), color='red')
    for i, txt in enumerate(annote4):
        plt.annotate(txt, (np.real(constellation4[i]), np.imag(constellation4[i]) + 0.05))
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.title('16-QAM')
    plt.grid()

    plt.tight_layout()
    plt.show()


def plot_error_rate(SNR_range, ser_PAM, ser_QPSK, ser_PSK, ser_QAM,
                    ber_PAM, ber_QPSK, ber_PSK, ber_QAM):

    plt.figure(figsize=(12, 12))

    plt.semilogy(SNR_range, ser_PAM, marker='o', label='PAM (M=8)')
    plt.semilogy(SNR_range, ser_QPSK, marker='o', label='QPSK (M=4)')
    plt.semilogy(SNR_range, ser_PSK, marker='o', label='PSK (M=8)')
    plt.semilogy(SNR_range, ser_QAM, marker='o', label='QAM (M=16)')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Symbol Error Rate')
    plt.title('Symbol Error Rate vs SNR')
    plt.grid(True)
    plt.legend()
    plt.show(block=False)
    plt.tight_layout()
    plt.figure(figsize=(12, 12))

    plt.semilogy(SNR_range, ber_PAM, marker='o', label='PAM (M=8)')
    plt.semilogy(SNR_range, ber_QPSK, marker='o', label='QPSK (M=4)')
    plt.semilogy(SNR_range, ber_PSK, marker='o', label='PSK (M=8)')
    plt.semilogy(SNR_range, ber_QAM, marker='o', label='QAM (M=16)')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate')
    plt.title('Bit Error Rate vs SNR')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show(block=True)


def plot_single_error_rate(SNR_range, mod, legend):
    min_val = 1e-12  # Small value to represent zero BER
    mod = np.maximum(mod, min_val)
    plt.figure(figsize=(12, 12))
    plt.semilogy(SNR_range, mod, marker='o', label=legend)
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate')
    plt.title('Bit Error Rate vs SNR')
    plt.grid(True)
    plt.legend()
    plt.show()


def show_constellation():

    PAM_list = constellation('PAM', 8)
    QPSK_list = constellation('PSK', 4)
    PSK8_list = constellation('PSK', 8)
    QAM_list = constellation('QAM', 16)

    GE_PAM = {"000": -7, "001": -5, "011": -3, "010": -1, "110": 1, "111": 3, "101": 5, "100": 7}
    GE_QPSK = {"00": QPSK_list[0], "01": QPSK_list[1], "11": QPSK_list[2], "10": QPSK_list[3]}
    GE_PSK8 = {"000": PSK8_list[0], "001": PSK8_list[1], "011": PSK8_list[2], "010": PSK8_list[3],
               "110": PSK8_list[4], "111": PSK8_list[5], "101": PSK8_list[6], "100": PSK8_list[7]}
    GE_QAM = {"0000": QAM_list[0], "0001": QAM_list[1], "0011": QAM_list[2], "0010": QAM_list[3],
              "0100": QAM_list[4], "0101": QAM_list[5], "0111": QAM_list[6], "0110": QAM_list[7],
              "1100": QAM_list[8], "1101": QAM_list[9], "1111": QAM_list[10], "1110": QAM_list[11],
              "1000": QAM_list[12], "1001": QAM_list[13], "1011": QAM_list[14], "1010": QAM_list[15]}

    PAM_text = GE_PAM.keys()
    QPSK_text = GE_QPSK.keys()
    PSK8_text = GE_PSK8.keys()
    QAM_text = GE_QAM.keys()

    plot_constellation(PAM_list, QPSK_list, PSK8_list, QAM_list,
                       PAM_text, QPSK_text, PSK8_text, QAM_text)


def modulation_diff():
    nr_symbols = 5*(10**5)
    PAM_list = constellation('PAM', 8)
    QPSK_list = constellation('PSK', 4)
    PSK8_list = constellation('PSK', 8)
    QAM_list = constellation('QAM', 16)

    GE_PAM = {"000": -7, "001": -5, "011": -3, "010": -1, "110": 1, "111": 3, "101": 5, "100": 7}
    GE_QPSK = {"00": QPSK_list[0], "01": QPSK_list[1], "11": QPSK_list[2], "10": QPSK_list[3]}
    GE_PSK8 = {"000": PSK8_list[0], "001": PSK8_list[1], "011": PSK8_list[2], "010": PSK8_list[3],
               "110": PSK8_list[4], "111": PSK8_list[5], "101": PSK8_list[6], "100": PSK8_list[7]}
    GE_QAM = {"0000": QAM_list[0], "0001": QAM_list[1], "0011": QAM_list[2], "0010": QAM_list[3],
              "0100": QAM_list[4], "0101": QAM_list[5], "0111": QAM_list[6], "0110": QAM_list[7],
              "1100": QAM_list[8], "1101": QAM_list[9], "1111": QAM_list[10], "1110": QAM_list[11],
              "1000": QAM_list[12], "1001": QAM_list[13], "1011": QAM_list[14], "1010": QAM_list[15]}

    SNR_db = np.linspace(0, 21, 44)
    PAM_ser, PAM_ber = [], []
    QPSK_ser, QPSK_ber = [], []
    PSK8_ser, PSK8_ber = [], []
    QAM_ser, QAM_ber = [], []

    for i, SNR in enumerate(SNR_db):
        print(SNR)
        PAM_symbols = np.random.choice(PAM_list, nr_symbols)
        QPSK_symbols = np.random.choice(QPSK_list, nr_symbols)
        PSK8_symbols = np.random.choice(PSK8_list, nr_symbols)
        QAM_symbols = np.random.choice(QAM_list, nr_symbols)

        PAM_rx = AWGN(PAM_symbols, SNR)
        QPSK_rx = AWGN(QPSK_symbols, SNR)
        PSK8_rx = AWGN(PSK8_symbols, SNR)
        QAM_rx = AWGN(QAM_symbols, SNR)

        _, PAM_detected_symbols = optimum_detector(PAM_rx, PAM_list)
        _, QPSK_detected_symbols = optimum_detector(QPSK_rx, QPSK_list)
        _, PSK8_detected_symbols = optimum_detector(PSK8_rx, PSK8_list)
        _, QAM_detected_symbols = optimum_detector(QAM_rx, QAM_list)

        PAM_ser.append(SER(PAM_symbols, PAM_detected_symbols))
        QPSK_ser.append(SER(QPSK_symbols, QPSK_detected_symbols))
        PSK8_ser.append(SER(PSK8_symbols, PSK8_detected_symbols))
        QAM_ser.append(SER(QAM_symbols, QAM_detected_symbols))

        PAM_ber.append(BER(PAM_symbols, PAM_detected_symbols, GE_PAM, 8))
        QPSK_ber.append(BER(QPSK_symbols, QPSK_detected_symbols, GE_QPSK, 4))
        PSK8_ber.append(BER(PSK8_symbols, PSK8_detected_symbols, GE_PSK8, 8))
        QAM_ber.append(BER(QAM_symbols, QAM_detected_symbols, GE_QAM, 16))

    plot_error_rate(SNR_db, PAM_ser, QPSK_ser, PSK8_ser, QAM_ser,
                    PAM_ber, QPSK_ber, PSK8_ber, QAM_ber)


def encoding_diff():
    print_debug = False
    plot_debug = False

    QPSK_list = constellation('PSK', 4)
    GE_QPSK = {"00": QPSK_list[0], "01": QPSK_list[1], "11": QPSK_list[2], "10": QPSK_list[3]}

    SNR_db = np.linspace(0, 21, 22)
    nr_symbols = 4*(10**6)

    plain_list = []
    encoded_list = []

    for i, SNR in enumerate(SNR_db):
        print(SNR)
        data = np.random.choice([0, 1], nr_symbols)
        encoded_data = hamming_encode(data)

        # Modulate data
        symbols_plain = np.array([GE_QPSK[''.join(map(str, data[i:i+2]))] for i in range(0, len(data), 2)])
        symbols_encoded = np.array([GE_QPSK[''.join(map(str, encoded_data[i:i+2]))] for i in range(0, len(encoded_data), 2)])

        rx_plain = AWGN(symbols_plain, SNR)
        rx_encoded = AWGN(symbols_encoded, SNR)

        detected_idx_plain, detected_symbols_plain = optimum_detector(rx_plain, QPSK_list)
        detected_idx_encoded, detected_symbols_encoded = optimum_detector(rx_encoded, QPSK_list)

        detected_bits_plain = symbols_to_bits(detected_symbols_plain, GE_QPSK)
        detected_bits_encoded = symbols_to_bits(detected_symbols_encoded, GE_QPSK)

        # Decode the detected binary data using Hamming decoder
        decoded_detected_bits = hamming_decode(detected_bits_encoded)

        # Calculate BER
        errors_plain = np.sum(detected_bits_plain != data)
        ber = errors_plain / (len(symbols_plain))

        errors_encoded = np.sum(decoded_detected_bits != data)
        ber_encoded = errors_encoded / (len(symbols_encoded))

        plain_list.append(ber)
        encoded_list.append(ber_encoded)

    plot_ber(SNR_db, encoded_list, plain_list)

    if print_debug == True:
        #print(f"Length encoded data: {len(encoded_data)}")
        #print(f"Length symbols encoded: {len(symbols_encoded)}")
        print(f"rx_symbols encoded: {rx_encoded}")
        #print(f"Length symbols detected encoded: {len(detected_symbols_encoded)}")
        #print(f"Length decoded data: {len(decoded_data)}")
        print(detected_idx_encoded)
        print(f"Plain BER: {ber_plain}")
        print(f"Encoded BER: {ber_encoded}")

    if plot_debug == True:
        plt.figure(figsize=(12, 12))
        plt.subplot(2, 2, 1)
        plt.scatter(np.real(symbols_plain), np.imag(symbols_plain), color='red', marker='x')
        plt.scatter(np.real(symbols_encoded), np.imag(symbols_encoded), color='blue', marker='o')
        plt.grid(True)
        plt.title("Transmitted symbols")

        plt.subplot(2, 2, 2)
        plt.scatter(np.real(rx_plain), np.imag(rx_plain), color='red', marker='x')
        plt.scatter(np.real(rx_encoded), np.imag(rx_encoded), color='blue', marker='o')
        plt.grid(True)
        plt.title("Received symbols")

        plt.subplot(2, 2, 3)
        plt.scatter(np.real(detected_symbols_plain), np.imag(detected_symbols_plain), color='red', marker='x')
        plt.scatter(np.real(detected_symbols_encoded), np.imag(detected_symbols_encoded), color='blue', marker='o')
        plt.grid(True)
        plt.title("Detected symbols")
        plt.show()


encoding_diff()
