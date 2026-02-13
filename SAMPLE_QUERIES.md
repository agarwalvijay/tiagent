# Sample Queries for TI Agent

These questions demonstrate the agent's ability to recommend across TI's diverse product portfolio.

## 🏭 System-Level Solutions (Multiple Product Types)

### Industrial IoT Gateway
```
I'm building an industrial IoT gateway that needs:
- Real-time processing for sensor data aggregation
- CAN-FD bus for industrial equipment communication
- Wireless connectivity for cloud upload
- Battery backup capability
What TI components should I use?
```
**Expected Products:** Sitara processor (AM62x), TCAN CAN-FD transceiver, CC wireless module, TPS power management

---

### Battery-Powered Environmental Monitor
```
Design a solar-powered air quality monitoring station that:
- Runs on coin cell + solar harvesting
- Measures temperature, humidity, CO2, particulates
- Sends data via LoRaWAN every 10 minutes
- Must last 5+ years on battery
What's the complete TI solution?
```
**Expected Products:** MSPM0 ultra-low-power MCU, TLV comparators (threshold detection), TPS power management, possibly CC wireless

---

### EV Battery Management System
```
I need components for an electric vehicle battery management system:
- Monitor 96 cells (voltage, current, temperature)
- High-precision voltage measurement (±0.1% accuracy)
- CAN-FD communication to vehicle ECU
- Automotive-grade (-40°C to 125°C)
- Functional safety requirements
Recommend the TI chip set.
```
**Expected Products:** C2000 MCU (F28/F29 series), TCAN CAN-FD transceiver, TLV precision comparators, TPS power rails

---

## 🔌 Interface & Communication Focus

### Multi-Protocol Industrial Controller
```
Build a factory automation controller supporting:
- IO-Link for sensor/actuator connectivity
- CAN bus for PLC communication
- USB for configuration and firmware updates
- Real-time performance for motion control
What TI products fit this application?
```
**Expected Products:** C2000 MCU (F28), TIOL IO-Link transceiver, TCAN CAN transceiver

---

### Automotive Camera Module
```
I'm designing an ADAS camera module that needs:
- High-performance image processing (1080p@60fps)
- MIPI CSI-2 camera interface
- CAN-FD for vehicle communication
- Low latency for safety-critical applications
Suggest TI processors and interface chips.
```
**Expected Products:** TDA4 Jacinto processor, TCAN CAN-FD transceiver

---

## ⚡ Power-Optimized Designs

### Wearable Health Monitor
```
Create a fitness tracker that:
- Tracks heart rate, SpO2, activity
- BLE 5.3 for smartphone sync
- 7-day battery life on 100mAh coin cell
- Always-on display with <100µA average current
Which TI chips minimize power consumption?
```
**Expected Products:** MSPM0 ultra-low-power MCU, CC wireless (BLE), TPS ultra-low-power LDO

---

### Smart Thermostat
```
Design a battery-powered smart thermostat:
- 2x AA batteries, 5-year life
- WiFi connectivity
- Temperature/humidity sensing with precision comparators
- HVAC control (relay drivers)
Recommend TI's lowest-power solution.
```
**Expected Products:** MSPM0 MCU, CC wireless (WiFi), TLV comparators, TPS power management

---

## 🚗 Automotive & Industrial

### Electric Power Steering (EPS) System
```
I need a motor control solution for electric power steering:
- Dual-core for safety redundancy
- CAN-FD communication
- High-resolution PWM for smooth control
- Automotive-grade, ASIL-D capable
What's the TI reference design?
```
**Expected Products:** C2000 F28 dual-core MCU, TCAN CAN-FD transceiver

---

### Building Automation Gateway
```
Build a smart building gateway with:
- BACnet/IP protocol support
- Multiple CAN buses for HVAC zones
- Ethernet connectivity
- Local HMI with touchscreen
Which TI processors and transceivers?
```
**Expected Products:** Sitara AM processor, TCAN CAN transceivers, possibly interface chips

---

## 🔬 Precision Measurement & Analog

### Portable Medical Device
```
Design a portable blood glucose meter:
- Ultra-precise analog front-end (16-bit ADC)
- Low-power MCU for long battery life
- BLE for smartphone data sync
- Medical-grade accuracy and safety
Suggest TI's most accurate solution.
```
**Expected Products:** MSPM0 MCU (with precision ADC), TLV precision comparators/op-amps, CC BLE, TPS ultra-low-noise LDO

---

### Industrial Process Monitor
```
I'm monitoring chemical processes with:
- 16 analog sensors (0-10V, 4-20mA)
- High-precision comparators for alarm thresholds
- Modbus RTU communication
- -40°C to 85°C industrial environment
What TI components provide the best accuracy?
```
**Expected Products:** C2000 or MSPM0 MCU, TLV precision comparators, interface transceivers

---

## 🌐 Wireless & Connectivity

### Smart Agriculture Sensor Node
```
Create a soil moisture sensor for precision farming:
- LoRaWAN communication (10km range)
- Solar + supercapacitor power
- Moisture, temperature, NPK sensors
- 10-year deployment, minimal maintenance
Which TI chips maximize range and battery life?
```
**Expected Products:** MSPM0 ultra-low-power MCU, CC sub-1GHz wireless, TPS power management, TLV comparators

---

### Smart City Parking Sensor
```
Design a parking space occupancy sensor:
- Ultrasonic or magnetometer sensing
- NB-IoT cellular connectivity
- 5+ year battery life (CR123A)
- Temperature: -30°C to 70°C
Recommend TI's most power-efficient solution.
```
**Expected Products:** MSPM0 MCU, TLV comparators (threshold detection), TPS power management

---

## 🏎️ High-Performance Computing

### Robotics Vision System
```
I'm building a warehouse robot that needs:
- Stereo vision processing (dual cameras)
- Deep learning inference for object detection
- Real-time path planning
- Multiple motor controllers
What TI processors can handle this?
```
**Expected Products:** TDA4 Jacinto AI processor, C2000 for motor control

---

### Edge AI Gateway
```
Design an industrial edge AI gateway:
- Run TensorFlow Lite models locally
- Aggregate data from 50+ Modbus sensors
- MQTT to cloud
- Automotive-grade for harsh environments
Which TI processor has the AI acceleration?
```
**Expected Products:** AM62P or TDA4 processor (with AI/NPU)

---

## 💡 Comparison & Trade-off Questions

### Low-Power MCU Comparison
```
Compare TI's ultra-low-power MCUs for a battery-powered sensor:
- MSPM0G3507 vs MSPM0L1306
- Which has lower standby current?
- Which has better ADC performance?
Show me the trade-offs with pricing.
```
**Expected Products:** Multiple MSPM0 variants with detailed comparison

---

### CAN Transceiver Selection
```
I need a CAN-FD transceiver for automotive use.
Compare TCAN1463A vs TCAN2410-Q1:
- Speed, power consumption, price
- Which supports higher bus voltages?
- Show datasheets and specifications.
```
**Expected Products:** TCAN series comparison

---

### Processor vs MCU Decision
```
Should I use a Sitara processor or C2000 MCU for:
- Motor control with 6 axes
- Ethernet connectivity
- HMI touchscreen
- Real-time control loops (<10µs)
Help me choose the right architecture.
```
**Expected Products:** Sitara AM vs C2000 F28 comparison

---

## 🎯 Specific Technical Challenges

### High-Speed Data Acquisition
```
I need to sample 8 analog channels at 1MSPS simultaneously:
- 12-bit resolution minimum
- Real-time FFT processing
- USB streaming to PC
Which TI MCU has the fastest ADC?
```
**Expected Products:** C2000 or high-performance MSPM0 with fast ADC

---

### Functional Safety Application
```
Design a safety-critical industrial controller:
- SIL-3 / IEC 61508 certification needed
- Dual-core lockstep for redundancy
- Built-in self-test (BIST)
- CAN-FD safety communication
What TI chips meet functional safety requirements?
```
**Expected Products:** C2000 F28 safety MCUs, TCAN-Q1 transceivers

---

## 📊 Usage Tips

### How to Use These Queries:

1. **Copy-paste** any question into the TI agent
2. The agent will search across **all 44 datasheets** covering:
   - MCUs (MSPM0, F28, F29, TMS320, MSP430)
   - Processors (Sitara AM, Jacinto TDA4)
   - Interface (TCAN, TIOL)
   - Wireless (CC series)
   - Comparators (TLV series)
   - Power Management (TPS series)

3. **Expect comprehensive responses** with:
   - Part number recommendations across multiple categories
   - Datasheet links (PDF/HTML)
   - Pricing information (when available)
   - Technical specifications
   - Trade-off analysis

### Custom Questions:

Feel free to modify these templates with your specific:
- **Application domain**: Industrial, automotive, medical, consumer
- **Constraints**: Power budget, cost target, size limits, temperature range
- **Requirements**: Communication protocols, processing power, memory needs
- **Compliance**: Automotive-grade (Q1), functional safety, medical certification
