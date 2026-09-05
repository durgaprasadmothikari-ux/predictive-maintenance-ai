"""
Humanoid Reliability Intelligence Engine - Dr. Nova, Chief Humanoid Reliability Engineer
Empowers the Predictive Maintenance AI with embodied conversational intelligence,
Web Speech API voice briefings, physics-grounded diagnostic narratives,
automated ISO-compliant work orders, and interactive telemetry triage.
"""

import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
import html


class HumanoidAssistant:
    """
    Embodied Humanoid Industrial Reliability Engineer Persona ('Dr. Nova').
    Provides conversational triage, text-to-speech audio briefings,
    first-person diagnostic assessments, and maintenance work orders.
    """

    NAME = "Dr. Nova"
    TITLE = "Chief Humanoid Reliability Engineer"
    SPECIALTY = "Autonomous Machinery Diagnostics & Plant Reliability"

    # =====================================================================
    # 1. FIRST-PERSON HUMAN-ENGINEER DIAGNOSTIC ASSESSMENTS
    # =====================================================================

    @classmethod
    def generate_humanoid_assessment(
        cls,
        machine_name: str,
        telemetry: Dict[str, Any],
        is_failure: bool,
        failure_prob: float,
        extra_metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generates a comprehensive first-person engineering assessment
        written in the voice of a seasoned industrial reliability engineer.
        """
        extra_metrics = extra_metrics or {}
        risk_pct = round(failure_prob * 100, 1)

        # Determine Priority & Triage Level
        if failure_prob >= 0.75:
            priority = "Priority 1 - Emergency Dispatch"
            urgency_badge = "CRITICAL / IMMEDIATE SHUTDOWN"
            triage_hours = "0 - 1 Hours"
            tone = "urgent"
        elif failure_prob >= 0.45:
            priority = "Priority 2 - High Urgency"
            urgency_badge = "ELEVATED RISK / INSPECT THIS SHIFT"
            triage_hours = "1 - 4 Hours"
            tone = "elevated"
        elif failure_prob >= 0.20:
            priority = "Priority 3 - Cautionary"
            urgency_badge = "OPERATING DEVIATION DETECTED"
            triage_hours = "Within 24 Hours"
            tone = "caution"
        else:
            priority = "Priority 4 - Routine Nominal"
            urgency_badge = "HEALTHY / OPTIMAL BASELINE"
            triage_hours = "Next Scheduled Maintenance"
            tone = "healthy"

        # Construct Machine-Specific Natural Voice Narrative & Physics Analysis
        if machine_name == "AI4I Industrial Machine":
            assessment = cls._assess_ai4i(telemetry, is_failure, failure_prob)
        elif machine_name == "Electric Motor":
            rul = extra_metrics.get("rul_hours", 2500.0)
            assessment = cls._assess_motor(telemetry, rul, failure_prob)
        elif machine_name == "Industrial Pump":
            assessment = cls._assess_pump(telemetry, is_failure, failure_prob)
        elif machine_name == "CNC Machine":
            assessment = cls._assess_cnc(telemetry, is_failure, failure_prob)
        elif machine_name == "Conveyor System":
            assessment = cls._assess_conveyor(telemetry, is_failure, failure_prob)
        else:
            assessment = cls._assess_generic(machine_name, telemetry, is_failure, failure_prob)

        assessment.update({
            "priority": priority,
            "urgency_badge": urgency_badge,
            "triage_hours": triage_hours,
            "risk_pct": risk_pct,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        return assessment

    # =====================================================================
    # MACHINE-SPECIFIC HUMAN NARRATIVES
    # =====================================================================

    @classmethod
    def _assess_ai4i(cls, t: Dict[str, Any], fail: bool, prob: float) -> Dict[str, Any]:
        air_t = t.get("Air temperature [K]", 300.0)
        proc_t = t.get("Process temperature [K]", 310.0)
        rpm = t.get("Rotational speed [rpm]", 1500.0)
        torque = t.get("Torque [Nm]", 40.0)
        wear = t.get("Tool wear [min]", 50.0)

        temp_diff = round(proc_t - air_t, 2)
        power_w = round(rpm * (2.0 * 3.14159 / 60.0) * torque, 1)
        overstrain = round(wear * torque, 1)

        findings = []
        hazards = []
        actions = []

        if temp_diff < 8.6 and rpm < 1380:
            findings.append(f"Heat Dissipation Breakdown: Process-to-air differential has compressed to {temp_diff} K under low speed ({rpm:.0f} RPM), impairing convective cooling.")
            hazards.append("Thermal burn hazard; spindle housing exceeding safe touch temperatures.")
            actions.append("Inspect coolant channels for blockage and verify heat exchanger blower airflow.")

        if torque > 55.0 and rpm < 1200:
            findings.append(f"Mechanical Overload: Excessive shaft torque ({torque:.1f} Nm) paired with speed droop suggests binding or severe mechanical resistance.")
            actions.append("Check workpiece clamp alignment, spindle drive belt tension, and motor drive current draw.")

        if wear > 180.0:
            findings.append(f"Critical Flank Tool Wear: Accumulated cutter engagement of {wear:.0f} minutes has reached the critical wear plateau.")
            hazards.append("Risk of explosive carbide tool chipping or catastrophic workpiece gouging.")
            actions.append("Initiate immediate tool-index replacement and inspect the tool holder taper.")

        if power_w > 8500.0 or (power_w < 3500.0 and rpm > 1400):
            findings.append(f"Electric Power Anomaly: Mechanical shaft power output ({power_w:.0f} W) deviates from nominal operational envelopes.")
            actions.append("Check drive inverter phase currents and test for transient load surges.")

        if not findings:
            findings.append(f"All parameters (temp differential {temp_diff} K, speed {rpm:.0f} RPM, torque {torque:.1f} Nm) are well within nominal steady-state envelopes.")
            actions.append("Continue standard production run; log readings in the shift ledger.")

        voice_script = (
            f"Hello operator, this is Engineer Nova. "
            + (f"Attention: I have flagged a critical failure risk of {prob*100:.1f} percent on this machine. " if prob > 0.5
               else f"I've completed my telemetry audit. Machine health is stable with failure probability at {prob*100:.1f} percent. ")
            + f"Thermal differential is measuring {temp_diff} Kelvin with torque at {torque:.1f} Newton meters. "
            + (f"My chief concern is {findings[0]} Please review the recommended actions before continuing." if prob > 0.3
               else "No immediate technician intervention is required at this time.")
        )

        return {
            "engineer_voice_script": voice_script,
            "diagnostic_narrative": (
                f"I've conducted a full diagnostic scan of the industrial machine telemetry. "
                f"Operating at {rpm:.0f} RPM and {torque:.1f} Nm torque, the mechanical power output is {power_w:.0f} W. "
                f"The thermal gradient between process and ambient is {temp_diff} K. "
                + " ".join(findings)
            ),
            "root_cause_hypothesis": (
                "Thermal boundary layer collapse combined with mechanical friction overloading" if temp_diff < 8.6 and torque > 50
                else "Progressive tool abrasive micro-fracturing and cutting edge degradation" if wear > 180
                else "Stochastic torque fluctuation within acceptable dynamic safety margins"
            ),
            "findings": findings,
            "safety_hazards": hazards if hazards else ["Standard rotating equipment pinch hazard; wear certified safety glasses."],
            "action_checklist": actions if actions else ["Maintain nominal feed and speed parameters.", "Monitor next tool cycle."],
            "required_tooling": ["Digital Infrared Thermometer", "Torque Wrench (0-100 Nm)", "Dial Indicator Gauge"] if prob > 0.3 else ["Visual Inspection Torch"]
        }

    @classmethod
    def _assess_motor(cls, t: Dict[str, Any], rul: float, prob: float) -> Dict[str, Any]:
        temp = t.get("Temperature", 65.0)
        curr = t.get("Output_Current", 25.0)
        volt = t.get("Output_Voltage", 400.0)
        speed = t.get("Speed", 1450.0)
        pwr = t.get("Power", 15.0)

        findings = []
        hazards = []
        actions = []

        if rul < 250.0:
            findings.append(f"Critical Remaining Useful Life: Estimated at only {rul:.0f} operating hours before expected stator/bearing failure.")
            hazards.append("Risk of stator winding dielectric breakdown and potential phase-to-ground flashover.")
            actions.append("Schedule motor swap-out during the immediate scheduled maintenance window.")

        if temp > 85.0:
            findings.append(f"High Stator Temperature: Winding temperature has escalated to {temp:.1f} °C (Class F insulation thermal ceiling approaches).")
            actions.append("Inspect cooling fan cowl for lint/debris clogging and check bearing grease condition.")

        if curr > 50.0:
            findings.append(f"Elevated Phase Draw: Motor is pulling {curr:.1f} Amperes, indicating severe mechanical overload or rotor eccentricity.")
            actions.append("Measure motor phase resistance balance with a micro-ohmmeter.")

        if not findings:
            findings.append(f"Stator thermals ({temp:.1f} °C) and supply current ({curr:.1f} A) remain well within the motor's nameplate specifications.")
            actions.append("Maintain routine vibration log; inspect terminal box seal during next PM.")

        voice_script = (
            f"Greetings technician, Dr. Nova here. I've analyzed your electric motor's health parameters. "
            + (f"Urgent alert: estimated remaining useful life has dropped to {rul:.0f} operating hours. " if rul < 500
               else f"The motor is running soundly with an estimated {rul:.0f} hours of remaining service life. ")
            + f"Current draw is {curr:.1f} Amps with stator core temperature at {temp:.1f} degrees Celsius. "
            + (f"Immediate action is required: {actions[0]}" if rul < 500 else "Parameters are well within nameplate tolerances.")
        )

        return {
            "engineer_voice_script": voice_script,
            "diagnostic_narrative": (
                f"My electromagnetic and thermal analysis indicates an estimated Remaining Useful Life of {rul:.1f} hours. "
                f"With the motor drawing {curr:.1f} A at {volt:.0f} V and operating at {speed:.0f} RPM, "
                + " ".join(findings)
            ),
            "root_cause_hypothesis": (
                "Thermal aging of Class F winding insulation accelerated by elevated stator load" if temp > 80
                else "Mechanical bearing race spalling causing localized parasitic drag" if rul < 500
                else "Normal thermal equilibrium under balanced electrical supply"
            ),
            "findings": findings,
            "safety_hazards": hazards if hazards else ["High voltage electrical hazard (400V+); observe proper NFPA 70E PPE."],
            "action_checklist": actions,
            "required_tooling": ["Megohmmeter (Insulation Resistance)", "Fluke Clamp Multimeter", "Acoustic Ultrasound Detector"]
        }

    @classmethod
    def _assess_pump(cls, t: Dict[str, Any], fail: bool, prob: float) -> Dict[str, Any]:
        flow = t.get("Flow_Rate", 250.0)
        suct = t.get("Suction_Pressure", 2.5)
        disc = t.get("Discharge_Pressure", 12.0)
        vibe = t.get("Vibration_RMS", 2.0)
        temp = t.get("Bearing_Temperature", 55.0)
        cav = t.get("Cavitation_Index", 2.0)

        head_diff = round(disc - suct, 2)
        findings = []
        hazards = []
        actions = []

        if cav < 1.0 or (suct < 1.5 and vibe > 4.5):
            findings.append(f"Active Cavitation Detected: Suction head of {suct:.2f} bar is insufficient, leading to vapor bubble collapse and micro-jet erosion.")
            hazards.append("Impeller vane pitting, rapid mechanical seal rupture, and pressurized fluid spray.")
            actions.append("Inspect suction strainer, confirm Net Positive Suction Head Available (NPSHa), and throttle discharge valve.")

        if vibe > 7.1:
            findings.append(f"ISO 10816-3 Zone D Severity: Vibration RMS of {vibe:.2f} mm/s indicates severe mechanical imbalance or coupling misalignment.")
            hazards.append("Catastrophic bearing cage fracture and casing split-ring blowout.")
            actions.append("Perform laser shaft alignment and check pump baseplate anchor bolt torque.")

        if temp > 85.0:
            findings.append(f"Bearing Thermal Runaway: Outboard bearing housing temperature reached {temp:.1f} °C.")
            actions.append("Check bearing oil sump level and inspect lubricant for water/particulate contamination.")

        if not findings:
            findings.append(f"Discharge head differential is steady at {head_diff:.2f} bar with smooth vibration ({vibe:.2f} mm/s) conforming to ISO Zone A.")
            actions.append("Maintain continuous vibration monitoring; check mechanical seal flush line.")

        voice_script = (
            f"Hello maintenance team, this is Dr. Nova with your industrial pump briefing. "
            + (f"Warning: pump failure risk is elevated at {prob*100:.1f} percent. " if prob > 0.4
               else f"The pump is operating normally with a failure risk of {prob*100:.1f} percent. ")
            + f"Total dynamic head is {head_diff:.2f} bar with vibration RMS measuring {vibe:.2f} millimeters per second. "
            + (f"Key finding: {findings[0]}" if prob > 0.4 else "Hydraulic performance is stable.")
        )

        return {
            "engineer_voice_script": voice_script,
            "diagnostic_narrative": (
                f"I've assessed the hydraulic and mechanical state of this pump. "
                f"Delivering {flow:.1f} L/min across a differential head of {head_diff:.2f} bar, "
                f"vibration stands at {vibe:.2f} mm/s RMS with bearing temperature at {temp:.1f} °C. "
                + " ".join(findings)
            ),
            "root_cause_hypothesis": (
                "Cavitation erosion on impeller blade suction eyes" if cav < 1.0
                else "Shaft angular misalignment or bearing cage degradation" if vibe > 6.0
                else "Hydrodynamic flow stability within optimal BEP (Best Efficiency Point)"
            ),
            "findings": findings,
            "safety_hazards": hazards if hazards else ["High pressure fluid lines; ensure eye protection and pressure relief verification."],
            "action_checklist": actions,
            "required_tooling": ["Laser Alignment Kit", "Pressure Test Manifold", "ISO Vibration Pen"]
        }

    @classmethod
    def _assess_cnc(cls, t: Dict[str, Any], fail: bool, prob: float) -> Dict[str, Any]:
        rpm = t.get("Spindle_Speed", 6000.0)
        feed = t.get("Feed_Rate", 800.0)
        force = t.get("Cutting_Force", 850.0)
        wear = t.get("Tool_Wear_Index", 50.0)
        vibe = t.get("Spindle_Vibration", 2.0)
        err = t.get("Axis_Feed_Error", 4.0)
        cool = t.get("Coolant_Pressure", 20.0)
        temp = t.get("Motor_Temperature", 45.0)

        findings = []
        hazards = []
        actions = []

        if vibe > 6.0:
            findings.append(f"Spindle Chatter Resonance: Vibration of {vibe:.2f} mm/s indicates dynamic cutting regenerative chatter or spindle bearing brinelling.")
            hazards.append("Spindle taper damage and high-velocity projectile carbide insert breakage.")
            actions.append("Reduce spindle speed by 10-15% to exit harmonic resonance; inspect spindle drawbar tension.")

        if cool < 10.0 and temp > 68.0:
            findings.append(f"Coolant Starvation: Pressure fell to {cool:.1f} bar while spindle motor rose to {temp:.1f} °C, accelerating thermal tool degradation.")
            actions.append("Check through-spindle coolant filter and clean high-pressure pump suction line.")

        if err > 16.0:
            findings.append(f"Servo Positioning Error: Axis feed error of {err:.2f} µm indicates ball screw backlash or linear encoder contamination.")
            actions.append("Clean linear glass scale and verify axis ball screw lubrication distribution.")

        if not findings:
            findings.append(f"Spindle harmonic vibration ({vibe:.2f} mm/s) and cutting force ({force:.1f} N) reflect clean chip shearing and stable tool path.")
            actions.append("Maintain cycle inspection; measure part tolerances on CMM.")

        voice_script = (
            f"Hello machinist, Engineer Nova here. CNC machining center status report: "
            + (f"Caution: failure probability is high at {prob*100:.1f} percent. " if prob > 0.4
               else f"The machine is cutting smoothly with failure probability under {prob*100:.1f} percent. ")
            + f"Spindle speed is {rpm:.0f} RPM with cutting force at {force:.1f} Newtons and vibration at {vibe:.2f} mm/s. "
            + (f"My assessment: {findings[0]}" if prob > 0.4 else "All axes and spindle thermals are optimal.")
        )

        return {
            "engineer_voice_script": voice_script,
            "diagnostic_narrative": (
                f"I've analyzed the multi-axis machining telemetry. "
                f"Operating at {rpm:.0f} RPM with feed rate of {feed:.0f} mm/min, cutting force is {force:.1f} N. "
                f"Axis positioning error is {err:.2f} µm. "
                + " ".join(findings)
            ),
            "root_cause_hypothesis": (
                "Self-excited regenerative tool chatter at spindle natural frequency" if vibe > 6.0
                else "Thermal expansion of ballscrew combined with insufficient coolant lubrication" if cool < 10.0
                else "Optimal chip load and balanced cutting dynamics"
            ),
            "findings": findings,
            "safety_hazards": hazards if hazards else ["High speed flying swarf and rotating cutter; keep enclosure door locked."],
            "action_checklist": actions,
            "required_tooling": ["Drawbar Force Gauge", "Dial Test Indicator (0.001mm)", "Laser Tool Setter"]
        }

    @classmethod
    def _assess_conveyor(cls, t: Dict[str, Any], fail: bool, prob: float) -> Dict[str, Any]:
        speed = t.get("Belt_Speed", 2.2)
        tension = t.get("Belt_Tension", 22.0)
        curr = t.get("Motor_Current", 45.0)
        temp = t.get("Roller_Bearing_Temperature", 48.0)
        vibe = t.get("Idler_Vibration", 2.0)
        slip = t.get("Belt_Slip_Percentage", 1.2)
        load = t.get("Load_Weight", 500.0)

        findings = []
        hazards = []
        actions = []

        if slip > 6.0:
            findings.append(f"Severe Belt Slip: Drive pulley slippage has reached {slip:.1f}%, causing friction heat and carcass rubber scuffing.")
            hazards.append("Friction fire hazard at the head drive pulley; belt carcass burnout.")
            actions.append("Check gravity take-up counterweight travel and inspect drive pulley rubber lagging.")

        if temp > 80.0 and vibe > 6.0:
            findings.append(f"Idler Bearing Seizure Imminent: Roller temperature is {temp:.1f} °C with {vibe:.2f} mm/s vibration, indicating grease depletion.")
            hazards.append("Frozen roller wearing through belt bottom cover; fire risk.")
            actions.append("Locate seized idler roll and replace roll assembly immediately.")

        if tension > 36.0 or tension < 9.0:
            status_txt = "Overtensioned (risk of splice tear)" if tension > 36.0 else "Undertensioned (sagging and belt derailment)"
            findings.append(f"Belt Tension Anomaly: Tension is {tension:.1f} kN — {status_txt}.")
            actions.append("Calibrate take-up winch position and inspect belt alignment switches.")

        if not findings:
            findings.append(f"Belt slip is low ({slip:.1f}%) and idler roller temperatures ({temp:.1f} °C) demonstrate smooth material transit.")
            actions.append("Verify belt skirtboard clearance and inspect tail pulley scraper.")

        voice_script = (
            f"Conveyor maintenance team, Engineer Nova reporting. "
            + (f"Attention: conveyor failure probability is elevated at {prob*100:.1f} percent. " if prob > 0.4
               else f"Bulk material transit is operating normally with low risk ({prob*100:.1f} percent). ")
            + f"Belt speed is {speed:.2f} meters per second with slip at {slip:.1f} percent. "
            + (f"Primary concern: {findings[0]}" if prob > 0.4 else "Drive and idlers are functioning smoothly.")
        )

        return {
            "engineer_voice_script": voice_script,
            "diagnostic_narrative": (
                f"I've completed an operational telemetry scan on the overland conveyor. "
                f"Conveying {load:.0f} t/h at {speed:.2f} m/s with {tension:.1f} kN tension, "
                f"the drive motor current is {curr:.1f} A. "
                + " ".join(findings)
            ),
            "root_cause_hypothesis": (
                "Drive pulley lagging glaze causing kinetic friction loss" if slip > 6.0
                else "Idler roller ball bearing failure due to particulate ingress" if temp > 80.0
                else "Balanced traction and optimal material distribution"
            ),
            "findings": findings,
            "safety_hazards": hazards if hazards else ["Nip point between belt and pulleys; strictly enforce Lockout/Tagout before approach."],
            "action_checklist": actions,
            "required_tooling": ["Infrared Pyrometer", "Tachometer Strobe", "Belt Tension Gauge"]
        }

    @classmethod
    def _assess_generic(cls, name: str, t: Dict[str, Any], fail: bool, prob: float) -> Dict[str, Any]:
        voice_script = f"Greetings operator, this is Dr. Nova. Assessment for {name}: failure probability is {prob*100:.1f} percent."
        return {
            "engineer_voice_script": voice_script,
            "diagnostic_narrative": f"I've inspected the telemetry stream for {name}. Failure probability stands at {prob*100:.1f}%.",
            "root_cause_hypothesis": "Normal wear equilibrium within baseline tolerances." if not fail else "Multiple operating variables deviating from standard envelope.",
            "findings": ["Operating parameters scanned."],
            "safety_hazards": ["Wear standard plant PPE (Hardhat, Safety Glasses, Steel-toe boots)."],
            "action_checklist": ["Maintain standard operating log."],
            "required_tooling": ["Standard Technician Toolset"]
        }

    # =====================================================================
    # 2. WEB SPEECH API / AUDIO BRIEFING SYNTHESIZER
    # =====================================================================

    @classmethod
    def render_voice_briefing_widget(cls, voice_text: str, element_id: str = "nova_speech") -> str:
        """
        Renders an interactive Text-to-Speech audio briefing bar using
        the browser-native Web Speech API (window.speechSynthesis).
        Runs 100% locally with zero cloud API latency or dependencies.
        """
        clean_text = html.escape(voice_text.replace("\n", " ").replace('"', "'"))
        unique_id = f"speech_{uuid.uuid4().hex[:8]}"

        html_code = f"""
        <div id="{unique_id}" style="
            background: linear-gradient(135deg, rgba(30,58,95,0.7), rgba(15,23,42,0.85));
            border: 1px solid rgba(96,165,250,0.35);
            border-radius: 14px;
            padding: 16px 20px;
            margin: 12px 0 20px 0;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 12px;
            box-shadow: 0 4px 18px rgba(0,0,0,0.25);
        ">
            <div style="display: flex; align-items: center; gap: 14px;">
                <div style="
                    width: 44px;
                    height: 44px;
                    border-radius: 50%;
                    background: linear-gradient(135deg, #3B82F6, #1D4ED8);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 22px;
                    box-shadow: 0 0 14px rgba(59,130,246,0.6);
                    position: relative;
                ">
                    🤖
                    <span style="
                        position: absolute;
                        bottom: 0;
                        right: 0;
                        width: 12px;
                        height: 12px;
                        background: #22C55E;
                        border-radius: 50%;
                        border: 2px solid #0F172A;
                    "></span>
                </div>
                <div>
                    <div style="font-weight: 700; color: #F8FAFC; font-size: 15px; letter-spacing: -0.2px;">
                        Dr. Nova's Audible Diagnostic Briefing
                    </div>
                    <div style="font-size: 12px; color: #94A3B8;">
                        Spoken audio synthesis · Web Speech API Engine
                    </div>
                </div>
            </div>

            <div style="display: flex; align-items: center; gap: 8px;">
                <button id="btn_play_{unique_id}" onclick="playNovaBriefing_{unique_id}()" style="
                    background: #2563EB;
                    color: #FFFFFF;
                    border: none;
                    border-radius: 8px;
                    padding: 9px 18px;
                    font-size: 13px;
                    font-weight: 600;
                    cursor: pointer;
                    display: flex;
                    align-items: center;
                    gap: 6px;
                    transition: all 0.2s;
                ">
                    🔊 Listen to Dr. Nova
                </button>
                <button id="btn_stop_{unique_id}" onclick="stopNovaBriefing_{unique_id}()" style="
                    background: rgba(255,255,255,0.08);
                    color: #CBD5E1;
                    border: 1px solid rgba(255,255,255,0.15);
                    border-radius: 8px;
                    padding: 9px 14px;
                    font-size: 13px;
                    font-weight: 500;
                    cursor: pointer;
                ">
                    ⏹ Stop
                </button>
                <span id="status_{unique_id}" style="font-size: 12px; color: #60A5FA; margin-left: 4px;"></span>
            </div>
        </div>

        <script>
        (function() {{
            const speechText = "{clean_text}";
            window.playNovaBriefing_{unique_id} = function() {{
                if (!('speechSynthesis' in window)) {{
                    alert('Web Speech API is not supported in this browser. Please use Chrome, Edge, or Firefox.');
                    return;
                }}
                window.speechSynthesis.cancel();
                const utterance = new SpeechSynthesisUtterance(speechText);
                utterance.rate = 1.05;
                utterance.pitch = 1.0;
                
                // Select a natural English voice if available
                const voices = window.speechSynthesis.getVoices();
                const preferredVoice = voices.find(v => (v.name.includes('Natural') || v.name.includes('Google') || v.name.includes('Neural') || v.name.includes('David') || v.name.includes('Samantha')) && v.lang.startsWith('en'));
                if (preferredVoice) {{
                    utterance.voice = preferredVoice;
                }}

                const statusEl = document.getElementById('status_{unique_id}');
                const playBtn = document.getElementById('btn_play_{unique_id}');

                utterance.onstart = function() {{
                    if (statusEl) statusEl.innerText = '🗣️ Speaking briefing...';
                    if (playBtn) playBtn.style.background = '#10B981';
                }};
                utterance.onend = function() {{
                    if (statusEl) statusEl.innerText = '✓ Briefing complete';
                    if (playBtn) playBtn.style.background = '#2563EB';
                }};
                utterance.onerror = function() {{
                    if (statusEl) statusEl.innerText = '';
                    if (playBtn) playBtn.style.background = '#2563EB';
                }};

                window.speechSynthesis.speak(utterance);
            }};

            window.stopNovaBriefing_{unique_id} = function() {{
                if ('speechSynthesis' in window) {{
                    window.speechSynthesis.cancel();
                }}
                const statusEl = document.getElementById('status_{unique_id}');
                const playBtn = document.getElementById('btn_play_{unique_id}');
                if (statusEl) statusEl.innerText = '';
                if (playBtn) playBtn.style.background = '#2563EB';
            }};
        }})();
        </script>
        """
        return html_code

    # =====================================================================
    # 3. INTERACTIVE CONVERSATIONAL INTELLIGENCE ("CHAT WITH DR. NOVA")
    # =====================================================================

    @classmethod
    def converse(
        cls,
        user_message: str,
        machine_name: str,
        telemetry: Dict[str, Any],
        assessment: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generates contextual conversational responses to questions from technicians
        in Dr. Nova's first-person humanoid voice.
        """
        msg = user_message.lower().strip()
        assessment = assessment or {}
        prob_pct = assessment.get("risk_pct", 50.0)
        findings = assessment.get("findings", ["Nominal operating parameters."])
        actions = assessment.get("action_checklist", ["Continue normal shift inspection."])
        hazards = assessment.get("safety_hazards", ["Standard machine guarding rules apply."])
        tools = assessment.get("required_tooling", ["Standard technician toolbox."])

        # 1. Plain English / Simple Explanation
        if any(k in msg for k in ["plain english", "explain", "simple", "what does this mean", "understand"]):
            return (
                f"Certainly! Here is my plain-language breakdown for your **{machine_name}**:\n\n"
                f"Right now, my reliability models calculate a **{prob_pct}% probability of mechanical failure**.\n\n"
                f"**What is physically happening inside the equipment:**\n"
                f"{findings[0]}\n\n"
                f"In simpler terms: think of this like driving a vehicle with low tire pressure at high speed. "
                f"While the machine is still turning, the internal stress is climbing rapidly. "
                f"If you continue running without intervention, the components will experience accelerated wear. "
                f"Here is the first thing I advise your crew to check: *{actions[0]}*"
            )

        # 2. Immediate Actions / What should I do right now?
        if any(k in msg for k in ["what should i do", "action", "steps", "immediate", "fix", "procedure", "how to"]):
            action_bullets = "\n".join([f"{i+1}. {a}" for i, a in enumerate(actions)])
            tool_bullets = ", ".join(tools)
            return (
                f"Here is your immediate operational gameplan, straight from my engineering log:\n\n"
                f"### Immediate Action Steps:\n"
                f"{action_bullets}\n\n"
                f"**Required Tooling & Equipment to Bring:**\n"
                f"🛠️ {tool_bullets}\n\n"
                f"**Urgency Window:** {assessment.get('triage_hours', 'Within 2-4 hours')}. "
                f"Make sure to sign off on the work order ticket before completing the shift."
            )

        # 3. Safety / Hazards / PPE / LOTO
        if any(k in msg for k in ["safe", "safety", "hazard", "ppe", "danger", "loto", "lockout"]):
            hazard_bullets = "\n".join([f"- ⚠️ {h}" for h in hazards])
            return (
                f"### 🛡️ Safety & Hazard Advisory from Dr. Nova\n\n"
                f"Before placing any hands or diagnostic tools on this **{machine_name}**, ensure full compliance with plant safety directives:\n\n"
                f"{hazard_bullets}\n\n"
                f"**Mandatory LOTO Protocol:**\n"
                f"1. Zero-Energy State Verification: De-energize 480V/400V main feeder breaker.\n"
                f"2. Bleed residual hydraulic/pneumatic stored pressure.\n"
                f"3. Apply personal safety padlock and danger tag at the disconnect switch.\n"
                f"4. Confirm zero rotation before removing any belt/spindle guards."
            )

        # 4. Work Order Request
        if any(k in msg for k in ["work order", "ticket", "dispatch", "job", "wo"]):
            return (
                f"I've pre-compiled an official maintenance dispatch work order for this machine! "
                f"You can view and download the complete ticket below in the **Maintenance Work Order** section, "
                f"complete with Asset Tag, Required Spare Parts, and Supervisor Sign-off blocks."
            )

        # 5. What-If Scenario / Parameter Tuning
        if any(k in msg for k in ["what if", "reduce", "lower", "increase", "speed", "temperature", "coolant"]):
            return (
                f"Great engineering question! In rotating machinery, parameter trade-offs are governed by dynamic stress laws:\n\n"
                f"- **Reducing Speed / Feed Rate by 15-20%**: Drops mechanical cutting power ($P = \\omega \\cdot \\tau$) proportionally, reducing frictional heat generation and lowering immediate thermal failure probability by approximately 25-35%.\n"
                f"- **Increasing Coolant Delivery**: Restores the thermal boundary layer, preventing rapid thermal tool wear.\n\n"
                f"You can use my **What-If Telemetry Simulator** below to slide any parameter and test my projected response live!"
            )

        # Fallback Natural Response
        return (
            f"I hear you loud and clear. As Chief Reliability Engineer on shift, my diagnostic assessment for the "
            f"**{machine_name}** indicates a **{prob_pct}% failure probability** ({assessment.get('priority', 'Caution')}).\n\n"
            f"**Key Engineering Takeaway:**\n"
            f"{assessment.get('diagnostic_narrative', 'Machine parameters are within normal baseline ranges.')}\n\n"
            f"Would you like me to detail the **Safety Hazards**, provide **Step-by-Step Triage Instructions**, or generate an **Official Work Order Ticket**?"
        )

    # =====================================================================
    # 4. ONE-CLICK ISO-14224 MAINTENANCE WORK ORDER GENERATOR
    # =====================================================================

    @classmethod
    def generate_work_order(
        cls,
        machine_name: str,
        telemetry: Dict[str, Any],
        is_failure: bool,
        failure_prob: float,
        assessment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generates an official maintenance work order ticket compliant with
        ISO-14224 / industrial asset management standards.
        """
        now = datetime.now()
        wo_id = f"WO-{now.strftime('%Y%m')}-{uuid.uuid4().hex[:6].upper()}"
        risk_pct = round(failure_prob * 100, 1)

        # Asset Code mapping
        asset_codes = {
            "AI4I Industrial Machine": "MILL-CNC-04",
            "Electric Motor": "MTR-IND-882",
            "Industrial Pump": "PUMP-CENT-101",
            "CNC Machine": "VMC-5AX-209",
            "Conveyor System": "CVR-OVERLAND-01",
            "Sensor Machine": "LINE-SENSOR-BAY-03"
        }
        asset_tag = asset_codes.get(machine_name, "ASSET-GEN-01")

        # Parts requisition
        parts_mapping = {
            "AI4I Industrial Machine": ["Carbide Insert Pack (ISO CNMG 120408)", "Spindle Heat Exchanger Filter", "Synthetic Way Lube ISO VG 68"],
            "Electric Motor": ["SKF Deep Groove Ball Bearing 6310-2Z/C3", "Mobil Polyrex EM Electric Motor Grease", "Winding Thermistor Lead Kit"],
            "Industrial Pump": ["Burgmann Mechanical Cartridge Seal H7N", "Viton O-Ring Casing Gasket Kit", "Stainless Impeller Wear Ring"],
            "CNC Machine": ["BT40 Spindle Pull Stud Set", "High-Pressure Coolant Nozzle Cluster", "THK Linear Guide Lubrication Cartridge"],
            "Conveyor System": ["Fenner Heavy-Duty Conveyor Belt Splice Kit", "CEMA C Idler Troughing Roll (127mm)", "Ceramic Pulley Lagging Strip"]
        }
        required_parts = parts_mapping.get(machine_name, ["Universal Gasket & Bearing Kit"])

        markdown_ticket = f"""# 📋 MAINTENANCE WORK ORDER
**Work Order ID:** `{wo_id}`  
**Asset Tag:** `{asset_tag}` ({machine_name})  
**Originating Engineer:** {cls.NAME} ({cls.TITLE})  
**Date / Timestamp:** {now.strftime('%Y-%m-%d %H:%M:%S')}  
**Triage Priority:** **{assessment.get('priority', 'Priority 2 - High Urgency')}**  
**Required Completion Window:** **{assessment.get('triage_hours', 'Within 4 Hours')}**

---

### 1. Equipment Diagnostic Summary
- **Current Failure Risk Score:** `{risk_pct}%`
- **Primary Failure Mechanism:** {assessment.get('root_cause_hypothesis', 'Mechanical stress deviation')}
- **Telemetry Anomaly Log:**
{chr(10).join(['  * ' + f for f in assessment.get('findings', ['Nominal operations.'])])}

---

### 2. Field Technician Action Checklist
{chr(10).join([f"{i+1}. [ ] {a}" for i, a in enumerate(assessment.get('action_checklist', ['Inspect machine.']))])}

---

### 3. Requisitioned Spare Parts & Tooling
- **Required Spare Parts:**
{chr(10).join(['  * ' + p for p in required_parts])}
- **Required Diagnostic Tooling:**
{chr(10).join(['  * ' + t for t in assessment.get('required_tooling', ['Standard Technician Kit'])])}

---

### 4. Safety & Lockout/Tagout (LOTO) Mandatory Directives
{chr(10).join(['- [ ] ' + h for h in assessment.get('safety_hazards', ['Wear PPE.'])])}
- [ ] De-energize and lock main circuit breaker with personal padlock.
- [ ] Dissipate all stored hydraulic, thermal, and pneumatic kinetic energy.
- [ ] Confirm zero RPM and post warning signage around perimeter.

---

### 5. Work Order Closeout Signatures
- **Assigned Technician Signature:** ___________________________  **Date:** ____________
- **Maintenance Supervisor Approval:** ___________________________  **Date:** ____________
- **AI Reliability Sign-off:** Verified by `{cls.NAME} Engine v2.4`
"""

        return {
            "wo_id": wo_id,
            "asset_tag": asset_tag,
            "priority": assessment.get("priority", "Priority 2"),
            "markdown_ticket": markdown_ticket,
            "parts": required_parts,
            "tools": assessment.get("required_tooling", []),
            "created_at": now.isoformat()
        }

    # =====================================================================
    # 5. STREAMLIT UI RENDERERS
    # =====================================================================

    @classmethod
    def render_command_center_briefing(cls, total_predictions: int, online_models: int):
        """Renders Dr. Nova's command center briefing hero card with audio briefing."""
        import streamlit as st

        briefing_text = (
            f"Welcome to the Reliability Command Center. I am Dr. Nova, your Chief Humanoid Reliability Engineer. "
            f"All {online_models} predictive models are online and monitoring telemetry streams. "
            f"Across our active session, {total_predictions} machinery assessments have been performed. "
            f"Select any machine workspace below to review live physics diagnostics or consult with me directly."
        )

        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, rgba(30,58,95,0.85), rgba(15,23,42,0.92));
                border: 1px solid rgba(96,165,250,0.3);
                border-radius: 18px;
                padding: 24px 26px;
                margin-bottom: 22px;
                box-shadow: 0 8px 24px rgba(0,0,0,0.25);
            ">
                <div style="display: flex; align-items: flex-start; justify-content: space-between; flex-wrap: wrap; gap: 16px;">
                    <div style="display: flex; align-items: center; gap: 16px;">
                        <div style="
                            width: 60px;
                            height: 60px;
                            border-radius: 50%;
                            background: linear-gradient(135deg, #2563EB, #1D4ED8);
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            font-size: 30px;
                            box-shadow: 0 0 20px rgba(37,99,235,0.7);
                            position: relative;
                        ">
                            🤖
                            <span style="
                                position: absolute;
                                bottom: 2px;
                                right: 2px;
                                width: 14px;
                                height: 14px;
                                background: #22C55E;
                                border-radius: 50%;
                                border: 2px solid #0F172A;
                            "></span>
                        </div>
                        <div>
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <span style="font-size: 20px; font-weight: 750; color: #F8FAFC;">Dr. Nova</span>
                                <span style="
                                    background: rgba(34,197,94,0.18);
                                    color: #4ADE80;
                                    border: 1px solid rgba(34,197,94,0.35);
                                    padding: 2px 10px;
                                    border-radius: 12px;
                                    font-size: 11px;
                                    font-weight: 700;
                                    text-transform: uppercase;
                                    letter-spacing: 0.08em;
                                ">Active · Online</span>
                            </div>
                            <div style="font-size: 13px; color: #94A3B8; margin-top: 2px;">
                                Chief Humanoid Reliability Engineer · Autonomous Machinery Diagnostics & Triage
                            </div>
                        </div>
                    </div>
                </div>
                <div style="
                    margin-top: 16px;
                    padding: 14px 18px;
                    border-radius: 12px;
                    background: rgba(255,255,255,0.035);
                    border: 1px solid rgba(255,255,255,0.08);
                    color: #E2E8F0;
                    font-size: 14px;
                    line-height: 1.6;
                ">
                    "{briefing_text}"
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.components.v1.html(cls.render_voice_briefing_widget(briefing_text, "cmd_speech"), height=95)

    @classmethod
    def render_humanoid_section(
        cls,
        machine_name: str,
        telemetry: Dict[str, Any],
        is_failure: bool,
        failure_prob: float,
        extra_metrics: Optional[Dict[str, Any]] = None,
        key_suffix: str = ""
    ):
        """
        Renders the full Humanoid AI Engineer experience on an active machine workspace:
        1. Audible voice briefing player
        2. First-person diagnostic assessment & root cause narrative
        3. Operational triage priority badge & checklist
        4. Safety hazards & mandatory LOTO protocols
        5. Interactive conversational chat with Dr. Nova
        6. One-click maintenance work order generation
        """
        import streamlit as st

        assessment = cls.generate_humanoid_assessment(
            machine_name=machine_name,
            telemetry=telemetry,
            is_failure=is_failure,
            failure_prob=failure_prob,
            extra_metrics=extra_metrics
        )

        st.divider()

        # Header with Persona
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; flex-wrap: wrap; gap: 10px;">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <span style="font-size: 26px;">🤖</span>
                    <div>
                        <div style="font-size: 18px; font-weight: 750; color: #F8FAFC;">
                            Dr. Nova's Humanoid Engineering Assessment
                        </div>
                        <div style="font-size: 12px; color: #94A3B8;">
                            Physical root-cause analysis, operational triage, and voice dispatch
                        </div>
                    </div>
                </div>
                <div style="
                    background: {'rgba(239,68,68,0.18)' if failure_prob >= 0.5 else 'rgba(34,197,94,0.18)'};
                    color: {'#F87171' if failure_prob >= 0.5 else '#4ADE80'};
                    border: 1px solid {'rgba(239,68,68,0.35)' if failure_prob >= 0.5 else 'rgba(34,197,94,0.35)'};
                    border-radius: 12px;
                    padding: 5px 14px;
                    font-size: 12px;
                    font-weight: 700;
                    letter-spacing: 0.05em;
                ">
                    {assessment['urgency_badge']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 1. Audible Speech Briefing Player
        st.components.v1.html(
            cls.render_voice_briefing_widget(assessment["engineer_voice_script"], f"speech_{key_suffix}"),
            height=95
        )

        # 2. Diagnostic Assessment & Root Cause in First-Person Voice
        tab_diag, tab_chat, tab_wo, tab_safety = st.tabs([
            "🧠 Dr. Nova's Assessment",
            "💬 Consult with Dr. Nova",
            "📋 Maintenance Work Order",
            "🛡️ Safety & LOTO Protocol"
        ])

        with tab_diag:
            st.markdown(
                f"""
                <div style="
                    background: rgba(255,255,255,0.03);
                    border: 1px solid rgba(148,163,184,0.18);
                    border-left: 4px solid {'#EF4444' if failure_prob >= 0.5 else '#22C55E'};
                    border-radius: 12px;
                    padding: 16px 20px;
                    margin-bottom: 16px;
                    line-height: 1.6;
                    color: #E2E8F0;
                ">
                    <div style="font-weight: 700; color: #93C5FD; font-size: 14px; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.06em;">
                        Engineer's Direct Telemetry Assessment
                    </div>
                    {assessment['diagnostic_narrative']}
                </div>
                """,
                unsafe_allow_html=True
            )

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 🔬 Root-Cause Mechanism")
                st.info(assessment["root_cause_hypothesis"])

                st.markdown("#### ⏱️ Dispatch Urgency")
                st.write(f"**Triage Tier:** {assessment['priority']}")
                st.write(f"**Recommended Window:** {assessment['triage_hours']}")

            with c2:
                st.markdown("#### 🛠️ Field Technician Action Checklist")
                for i, act in enumerate(assessment["action_checklist"]):
                    st.checkbox(act, key=f"act_{key_suffix}_{i}", value=False)

                st.markdown("#### 🧰 Required Diagnostic Tooling")
                st.write(", ".join(assessment["required_tooling"]))

        with tab_chat:
            st.markdown("#### 💬 Ask Dr. Nova About This Machine")
            st.caption("Ask questions about this machine's vibration, temperature, safety, or what-if operational changes.")

            # Quick Prompt Buttons
            q_cols = st.columns(4)
            quick_question = None
            if q_cols[0].button("💬 Plain English", key=f"q_plain_{key_suffix}", use_container_width=True):
                quick_question = "Explain this failure risk in plain English for an apprentice."
            if q_cols[1].button("🛠️ Immediate Steps", key=f"q_steps_{key_suffix}", use_container_width=True):
                quick_question = "What exact steps should my maintenance crew take right now?"
            if q_cols[2].button("🛡️ Safety Directives", key=f"q_safe_{key_suffix}", use_container_width=True):
                quick_question = "What are the critical safety hazards before approaching this machine?"
            if q_cols[3].button("🧪 What-If Tuning", key=f"q_whatif_{key_suffix}", use_container_width=True):
                quick_question = "What if we reduce operating speed or feed rate by 15%?"

            chat_input_val = st.text_input(
                "Type a message to Dr. Nova:",
                value=quick_question or "",
                placeholder="e.g. Nova, why is the torque high while rotational speed is nominal?",
                key=f"user_chat_{key_suffix}"
            )

            if chat_input_val:
                response = cls.converse(
                    user_message=chat_input_val,
                    machine_name=machine_name,
                    telemetry=telemetry,
                    assessment=assessment
                )
                st.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(135deg, rgba(30,58,95,0.5), rgba(15,23,42,0.6));
                        border: 1px solid rgba(59,130,246,0.3);
                        border-radius: 12px;
                        padding: 16px 18px;
                        margin-top: 12px;
                        color: #E2E8F0;
                        line-height: 1.6;
                    ">
                        <div style="font-weight: 700; color: #60A5FA; margin-bottom: 8px; display: flex; align-items: center; gap: 6px;">
                            🤖 Dr. Nova's Reply:
                        </div>
                        {response}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with tab_wo:
            st.markdown("#### 📋 Official Maintenance Work Order Dispatch")
            st.caption("ISO-14224 compliant work ticket with parts requisition and supervisor sign-off.")

            wo_data = cls.generate_work_order(
                machine_name=machine_name,
                telemetry=telemetry,
                is_failure=is_failure,
                failure_prob=failure_prob,
                assessment=assessment
            )

            st.text_area(
                "Work Order Ticket (Markdown):",
                value=wo_data["markdown_ticket"],
                height=300,
                key=f"wo_text_{key_suffix}"
            )

            d1, d2 = st.columns(2)
            with d1:
                st.download_button(
                    label="📥 Download Work Order (.md)",
                    data=wo_data["markdown_ticket"],
                    file_name=f"{wo_data['wo_id']}_{machine_name.replace(' ', '_')}.md",
                    mime="text/markdown",
                    key=f"dl_wo_{key_suffix}",
                    use_container_width=True
                )
            with d2:
                st.success(f"Work Order Generated: `{wo_data['wo_id']}` · Tag `{wo_data['asset_tag']}`")

        with tab_safety:
            st.markdown("#### 🛡️ Plant Safety & Lockout/Tagout Directives")
            for h in assessment["safety_hazards"]:
                st.warning(f"⚠️ {h}")

            st.markdown("#### 🔒 Lockout/Tagout (LOTO) Protocol")
            st.write("1. **Zero-Energy State:** De-energize 400V/480V supply at main motor control center.")
            st.write("2. **Residual Energy Dissipation:** Bleed hydraulic, thermal, and mechanical spring tension.")
            st.write("3. **Lock & Tag:** Affix personal OSHA/ISO lockout padlock and warning tag.")
            st.write("4. **Zero Motion Check:** Verify complete zero RPM before removing belt, coupling, or spindle guards.")

