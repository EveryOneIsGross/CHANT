import argparse
import json
import numpy as np
import scipy.io.wavfile as wav
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Callable, Optional, Tuple
from enum import Enum
from scipy.signal import butter, lfilter, sosfilt, sosfilt_zi

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


class PhonemeType(Enum):
    VOWEL = 'vowel'
    NASAL = 'nasal'
    FRICATIVE = 'fricative'
    PLOSIVE = 'plosive'
    SILENCE = 'silence'


@dataclass
class FormantSpec:
    fc: float
    bw: float
    amp: float


@dataclass
class PhonemeSpec:
    phoneme_type: PhonemeType
    formants: List[Tuple[float, float, float]]
    antiformants: List[Tuple[float, float]] = field(default_factory=list)
    burst_freq: float = 0.0
    burst_bw: float = 0.0
    aspiration: float = 0.0
    voicing: float = 1.0
    skirt_width: float = 1.0
    glottal_tension: float = 0.5


def load_phoneme_table(path: Optional[Path] = None) -> Dict[str, PhonemeSpec]:
    table_path = Path(path) if path else Path(__file__).with_name("phonemes.yaml")
    def _fallback() -> Dict[str, PhonemeSpec]:
        # Minimal inline fallback if phonemes.yaml is missing
        base = {
            # Vowels
            'AA': PhonemeSpec(PhonemeType.VOWEL, [(700, 130, 0), (1220, 70, -6), (2600, 160, -24), (3300, 250, -28)], voicing=1.0),
            'AH': PhonemeSpec(PhonemeType.VOWEL, [(640, 120, 0), (1190, 80, -5), (2390, 140, -22), (3300, 250, -28)], voicing=1.0),
            'IY': PhonemeSpec(PhonemeType.VOWEL, [(270, 60, 0), (2290, 90, -16), (3010, 150, -22), (3500, 200, -30)], voicing=1.0),
            'UW': PhonemeSpec(PhonemeType.VOWEL, [(300, 80, 0), (870, 80, -10), (2240, 120, -26), (3300, 250, -30)], voicing=1.0),
            'EH': PhonemeSpec(PhonemeType.VOWEL, [(530, 100, 0), (1840, 90, -5), (2480, 140, -18), (3300, 250, -28)], voicing=1.0),
            'AE': PhonemeSpec(PhonemeType.VOWEL, [(660, 120, 0), (1720, 90, -4), (2410, 150, -20), (3300, 250, -28)], voicing=1.0),
            'AO': PhonemeSpec(PhonemeType.VOWEL, [(570, 100, 0), (840, 80, -5), (2410, 130, -26), (3300, 250, -28)], voicing=1.0),
            # Nasal + simple consonants
            'M': PhonemeSpec(PhonemeType.NASAL, [(250, 60, -3), (1000, 100, -20), (2000, 150, -35)], antiformants=[(750, 80)], voicing=0.9),
            'N': PhonemeSpec(PhonemeType.NASAL, [(250, 60, -3), (1200, 100, -20), (2200, 150, -35)], antiformants=[(1000, 90)], voicing=0.9),
            'NG': PhonemeSpec(PhonemeType.NASAL, [(250, 60, -3), (1100, 100, -22), (2100, 150, -38)], antiformants=[(800, 85)], voicing=0.9),
            'P': PhonemeSpec(PhonemeType.PLOSIVE, [(200, 400, -20), (800, 300, -25), (1800, 400, -30)], burst_freq=400, burst_bw=600, aspiration=0.12, voicing=0.0),
            'T': PhonemeSpec(PhonemeType.PLOSIVE, [(200, 400, -20), (1200, 300, -22), (2500, 400, -28)], burst_freq=3000, burst_bw=1500, aspiration=0.18, voicing=0.0),
            'K': PhonemeSpec(PhonemeType.PLOSIVE, [(200, 400, -20), (1000, 300, -20), (2200, 400, -25)], burst_freq=1500, burst_bw=1000, aspiration=0.14, voicing=0.0),
            'B': PhonemeSpec(PhonemeType.PLOSIVE, [(200, 300, -15), (800, 250, -22), (1800, 350, -28)], burst_freq=350, burst_bw=500, aspiration=0.05, voicing=0.3),
            'D': PhonemeSpec(PhonemeType.PLOSIVE, [(200, 350, -15), (1200, 280, -20), (2500, 380, -26)], burst_freq=2800, burst_bw=1200, aspiration=0.08, voicing=0.35),
            'G': PhonemeSpec(PhonemeType.PLOSIVE, [(200, 350, -15), (1000, 280, -18), (2200, 380, -23)], burst_freq=1400, burst_bw=800, aspiration=0.06, voicing=0.32),
            'S': PhonemeSpec(PhonemeType.FRICATIVE, [(4000, 800, -18), (5500, 900, -20), (7000, 1000, -25)], aspiration=0.92, voicing=0.0),
            'Z': PhonemeSpec(PhonemeType.FRICATIVE, [(4000, 700, -20), (5500, 800, -22), (7000, 900, -27)], aspiration=0.7, voicing=0.4),
            'F': PhonemeSpec(PhonemeType.FRICATIVE, [(3000, 1200, -22), (5000, 1500, -25), (7000, 1500, -30)], aspiration=0.88, voicing=0.0),
            'V': PhonemeSpec(PhonemeType.FRICATIVE, [(3000, 1100, -24), (5000, 1400, -27), (7000, 1400, -32)], aspiration=0.6, voicing=0.45),
            'TH': PhonemeSpec(PhonemeType.FRICATIVE, [(4500, 1500, -25), (6000, 1500, -28), (7500, 1500, -33)], aspiration=0.82, voicing=0.0),
            'DH': PhonemeSpec(PhonemeType.FRICATIVE, [(4500, 1400, -27), (6000, 1400, -30), (7500, 1400, -35)], aspiration=0.55, voicing=0.5),
            'SH': PhonemeSpec(PhonemeType.FRICATIVE, [(2500, 600, -15), (4000, 800, -18), (6000, 1000, -25)], aspiration=0.9, voicing=0.0),
            'ZH': PhonemeSpec(PhonemeType.FRICATIVE, [(2500, 550, -18), (4000, 750, -21), (6000, 950, -28)], aspiration=0.65, voicing=0.42),
            'H': PhonemeSpec(PhonemeType.FRICATIVE, [(500, 400, -10), (1500, 500, -15), (2500, 600, -22)], aspiration=0.75, voicing=0.0),
            'CH': PhonemeSpec(PhonemeType.PLOSIVE, [(2500, 600, -18), (4000, 800, -22), (6000, 1000, -28)], burst_freq=4000, burst_bw=2000, aspiration=0.25, voicing=0.0),
            'JH': PhonemeSpec(PhonemeType.PLOSIVE, [(2500, 550, -20), (4000, 750, -24), (6000, 950, -30)], burst_freq=3500, burst_bw=1800, aspiration=0.18, voicing=0.38),
            'L': PhonemeSpec(PhonemeType.NASAL, [(350, 70, -3), (1100, 90, -18), (2400, 130, -30)], antiformants=[(900, 70)], voicing=0.95),
            'R': PhonemeSpec(PhonemeType.NASAL, [(420, 80, -4), (1300, 100, -16), (1600, 120, -22)], antiformants=[(1100, 80)], voicing=0.92),
            'W': PhonemeSpec(PhonemeType.NASAL, [(300, 70, -3), (800, 80, -12), (2300, 130, -28)], voicing=0.95),
            'Y': PhonemeSpec(PhonemeType.NASAL, [(280, 60, -3), (2200, 90, -14), (2900, 120, -22)], voicing=0.95),
            'SIL': PhonemeSpec(PhonemeType.SILENCE, [(0, 100, -100), (0, 100, -100), (0, 100, -100)], voicing=0.0),
        }
        return base

    try:
        raw_text = table_path.read_text()
        if yaml is not None:
            data = yaml.safe_load(raw_text)
        else:
            data = json.loads(raw_text)
        table: Dict[str, PhonemeSpec] = {}
        for key, entry in data.items():
            ptype = PhonemeType(entry.get('phoneme_type', PhonemeType.VOWEL.value))
            formants = [tuple(fmt) for fmt in entry.get('formants', [])]
            antiformants = [tuple(af) for af in entry.get('antiformants', [])]
            table[key] = PhonemeSpec(
                phoneme_type=ptype,
                formants=formants,
                antiformants=antiformants,
                burst_freq=entry.get('burst_freq', 0.0),
                burst_bw=entry.get('burst_bw', 0.0),
                aspiration=entry.get('aspiration', 0.0),
                voicing=entry.get('voicing', 1.0),
                skirt_width=entry.get('skirt_width', 1.0),
                glottal_tension=entry.get('glottal_tension', 0.5)
            )
        if table:
            return table
    except Exception:
        pass
    return _fallback()


PHONEME_TABLE = load_phoneme_table()


DIGRAPH_MAP = {
    'th': 'TH', 'sh': 'SH', 'ch': 'CH', 'ng': 'NG', 'zh': 'ZH', 'dh': 'DH', 'jh': 'JH',
    'wh': 'W', 'ph': 'F', 'gh': 'G', 'ck': 'K', 'qu': 'K'
}


SIMPLE_DICT = {
    'a': 'AH', 'b': 'B', 'c': 'K', 'd': 'D', 'e': 'EH', 'f': 'F', 'g': 'G',
    'h': 'H', 'i': 'IY', 'j': 'JH', 'k': 'K', 'l': 'L', 'm': 'M', 'n': 'N',
    'o': 'AO', 'p': 'P', 'q': 'K', 'r': 'R', 's': 'S', 't': 'T', 'u': 'UW',
    'v': 'V', 'w': 'W', 'x': 'K', 'y': 'Y', 'z': 'Z', ' ': 'SIL'
}


class JitterGenerator:
    def __init__(self, sr):
        self.sr = sr
        self.periods = [0.05, 0.111, 1.219]
        self.targets = [0.0, 0.0, 0.0]
        self.current = [0.0, 0.0, 0.0]
        self.counters = [0, 0, 0]
        self.max_deviation = 0.015
        self.slow = 0.0

    def next(self) -> float:
        total = 0.0
        for j in range(3):
            period_samples = int(self.periods[j] * self.sr)
            self.counters[j] += 1
            if self.counters[j] >= period_samples:
                self.counters[j] = 0
                self.targets[j] = np.random.uniform(-self.max_deviation / 3, self.max_deviation / 3)
            alpha = 1.0 / (period_samples * 0.1 + 1)
            self.current[j] += alpha * (self.targets[j] - self.current[j])
            total += self.current[j]
        self.slow += 0.001 * (total - self.slow)
        return self.slow


class RegisterModel:
    def __init__(self, voice_type='tenor'):
        self.voice_type = voice_type
        configs = {
            'soprano': (440.0, 500, 1200),
            'alto': (330.0, 450, 1100),
            'tenor': (220.0, 400, 1000),
            'bass': (165.0, 350, 900)
        }
        self.register_center, self.f1_threshold, self.f2_threshold = configs.get(voice_type, configs['tenor'])

    def shift_formants(self, formants: List[Dict], f0: float, effort: float) -> List[Dict]:
        shifted = []
        register_ratio = f0 / self.register_center if self.register_center > 0 else 1.0
        for i, fmt in enumerate(formants):
            fc = fmt['fc']
            bw = fmt['bw']
            amp = fmt['amp']
            skirt = fmt.get('skirt', 1.0)
            if i == 0 and fc > 0 and fc < self.f1_threshold and f0 > 0:
                harmonic = max(1, round(fc / f0))
                target_fc = harmonic * f0
                if abs(target_fc - fc) < fc * 0.35:
                    fc = fc * 0.55 + target_fc * 0.45
            if i == 1 and fc > 0 and fc < self.f2_threshold and f0 > 0:
                harmonic = max(2, round(fc / f0))
                target_fc = harmonic * f0
                if abs(target_fc - fc) < fc * 0.28:
                    fc = fc * 0.65 + target_fc * 0.35
            if register_ratio > 1.5:
                fc *= 1.0 + (register_ratio - 1.5) * 0.06 * (i + 1) * 0.25
            if i > 0:
                amp *= 1.0 + effort * 0.28 * i
            if register_ratio > 1.2 and i > 1:
                amp *= 1.0 + (register_ratio - 1.2) * 0.12
            bw *= 1.0 + effort * 0.08
            shifted.append({'fc': fc, 'bw': bw, 'amp': amp, 'skirt': skirt})
        return shifted


@dataclass
class FORMESEvent:
    start_time: float
    duration: float
    event_type: str
    params: Dict = field(default_factory=dict)
    children: List['FORMESEvent'] = field(default_factory=list)
    transforms: List[Callable] = field(default_factory=list)


class FORMESProcess:
    def __init__(self, name: str):
        self.name = name
        self.rules: List[Callable] = []
        self.state: Dict = {}

    def add_rule(self, rule: Callable):
        self.rules.append(rule)

    def apply(self, event: FORMESEvent, context: Dict) -> FORMESEvent:
        result = event
        for rule in self.rules:
            result = rule(result, context, self.state)
        return result


class FORMESScheduler:
    def __init__(self):
        self.processes: Dict[str, FORMESProcess] = {}
        self.global_transforms: List[Callable] = []
        self.context: Dict = {'time': 0.0, 'tempo': 1.0, 'dynamics': 0.7}

    def register_process(self, process: FORMESProcess):
        self.processes[process.name] = process

    def add_global_transform(self, transform: Callable):
        self.global_transforms.append(transform)

    def schedule(self, events: List[FORMESEvent]) -> List[FORMESEvent]:
        scheduled = []
        for event in events:
            processed = event
            if event.event_type in self.processes:
                processed = self.processes[event.event_type].apply(event, self.context)
            for transform in self.global_transforms:
                processed = transform(processed, self.context)
            for child_transform in processed.transforms:
                processed = child_transform(processed, self.context)
            scheduled.append(processed)
            if processed.children:
                child_scheduled = self.schedule(processed.children)
                scheduled.extend(child_scheduled)
        return scheduled


def rule_phrase_dynamics(event: FORMESEvent, context: Dict, state: Dict) -> FORMESEvent:
    if 'phrase_position' in event.params:
        pos = event.params['phrase_position']
        arch = np.sin(np.pi * pos)
        event.params['effort'] = event.params.get('effort', 0.5) * (0.7 + 0.5 * arch)
        event.params['vib_depth'] = event.params.get('vib_depth', 0.01) * (0.5 + arch)
    return event


def rule_legato_connection(event: FORMESEvent, context: Dict, state: Dict) -> FORMESEvent:
    if 'prev_phoneme_type' in state and event.params.get('phoneme_type') == PhonemeType.VOWEL:
        prev = state['prev_phoneme_type']
        if prev in [PhonemeType.VOWEL, PhonemeType.NASAL]:
            event.params['attack_time'] = event.params.get('attack_time', 0.02) * 1.5
            event.params['transition_smooth'] = 0.85
    state['prev_phoneme_type'] = event.params.get('phoneme_type')
    return event


def rule_stress_accent(event: FORMESEvent, context: Dict, state: Dict) -> FORMESEvent:
    if event.params.get('stressed', False):
        event.params['effort'] = event.params.get('effort', 0.5) * 1.22
        event.params['f0_offset'] = event.params.get('f0_offset', 0) + 0.025
        event.duration *= 1.12
    return event


def rule_final_lengthening(event: FORMESEvent, context: Dict, state: Dict) -> FORMESEvent:
    if event.params.get('phrase_final', False):
        event.duration *= 1.35
        event.params['effort'] = event.params.get('effort', 0.5) * 0.88
    return event


def transform_tempo(event: FORMESEvent, context: Dict) -> FORMESEvent:
    tempo = context.get('tempo', 1.0)
    event.duration /= tempo
    event.start_time /= tempo
    return event


def transform_dynamics(event: FORMESEvent, context: Dict) -> FORMESEvent:
    dyn = context.get('dynamics', 0.7)
    event.params['effort'] = event.params.get('effort', 0.5) * dyn
    return event


def create_formes_system() -> FORMESScheduler:
    scheduler = FORMESScheduler()
    phrase_process = FORMESProcess('phrase')
    phrase_process.add_rule(rule_phrase_dynamics)
    phrase_process.add_rule(rule_final_lengthening)
    scheduler.register_process(phrase_process)
    phoneme_process = FORMESProcess('phoneme')
    phoneme_process.add_rule(rule_legato_connection)
    phoneme_process.add_rule(rule_stress_accent)
    scheduler.register_process(phoneme_process)
    scheduler.add_global_transform(transform_tempo)
    scheduler.add_global_transform(transform_dynamics)
    return scheduler


class CoarticulationEngine:
    def __init__(self, lookahead_frames: int = 2, lookbehind_frames: int = 1):
        self.lookahead = lookahead_frames
        self.lookbehind = lookbehind_frames
        self.coartic_window = 0.04
        self.formant_influence = {
            PhonemeType.VOWEL: 0.6,
            PhonemeType.NASAL: 0.45,
            PhonemeType.FRICATIVE: 0.25,
            PhonemeType.PLOSIVE: 0.15,
            PhonemeType.SILENCE: 0.0
        }

    def apply_coarticulation(self, sparsegram: List[Dict]) -> List[Dict]:
        if len(sparsegram) < 2:
            return sparsegram
        result = []
        for i, frame in enumerate(sparsegram):
            new_frame = self._deep_copy_frame(frame)
            context_frames = []
            weights = []
            for j in range(max(0, i - self.lookbehind), i):
                dist = i - j
                weight = 0.3 / dist
                context_frames.append(sparsegram[j])
                weights.append(weight)
            for j in range(i + 1, min(len(sparsegram), i + 1 + self.lookahead)):
                dist = j - i
                weight = 0.5 / dist
                context_frames.append(sparsegram[j])
                weights.append(weight)
            if context_frames:
                new_frame = self._blend_formants(new_frame, context_frames, weights)
                new_frame = self._compute_transitions(new_frame, sparsegram, i)
            result.append(new_frame)
        return result

    def _deep_copy_frame(self, frame: Dict) -> Dict:
        new_frame = frame.copy()
        new_frame['formants'] = [f.copy() for f in frame['formants']]
        if 'antiformants' in frame:
            new_frame['antiformants'] = [a.copy() for a in frame['antiformants']]
        if 'noise_formants' in frame:
            new_frame['noise_formants'] = [n.copy() for n in frame['noise_formants']]
        return new_frame

    def _blend_formants(self, frame: Dict, context: List[Dict], weights: List[float]) -> Dict:
        current_type = frame.get('phoneme_type', PhonemeType.VOWEL)
        current_influence = self.formant_influence.get(current_type, 0.3)
        if current_influence < 0.1:
            return frame
        total_weight = sum(weights)
        if total_weight < 0.01:
            return frame
        for i, fmt in enumerate(frame['formants']):
            if fmt['fc'] < 10:
                continue
            fc_shift = 0.0
            bw_shift = 0.0
            for ctx, w in zip(context, weights):
                ctx_type = ctx.get('phoneme_type', PhonemeType.SILENCE)
                ctx_influence = self.formant_influence.get(ctx_type, 0.0)
                if i < len(ctx['formants']) and ctx['formants'][i]['fc'] > 10:
                    ctx_fmt = ctx['formants'][i]
                    blend = w * ctx_influence * current_influence / total_weight
                    fc_shift += (ctx_fmt['fc'] - fmt['fc']) * blend * 0.15
                    bw_shift += (ctx_fmt['bw'] - fmt['bw']) * blend * 0.1
            fmt['fc'] = max(20, fmt['fc'] + fc_shift)
            fmt['bw'] = max(10, fmt['bw'] + bw_shift)
        return frame

    def _compute_transitions(self, frame: Dict, sparsegram: List[Dict], idx: int) -> Dict:
        if idx < len(sparsegram) - 1:
            next_frame = sparsegram[idx + 1]
            next_type = next_frame.get('phoneme_type', PhonemeType.SILENCE)
            curr_type = frame.get('phoneme_type', PhonemeType.VOWEL)
            if curr_type == PhonemeType.VOWEL and next_type in [PhonemeType.PLOSIVE, PhonemeType.FRICATIVE]:
                frame['transition_type'] = 'vc_closure'
                frame['closure_target'] = next_frame.get('phoneme', 'SIL')
            elif curr_type in [PhonemeType.PLOSIVE, PhonemeType.NASAL] and next_type == PhonemeType.VOWEL:
                frame['transition_type'] = 'cv_release'
                frame['release_target'] = next_frame.get('phoneme', 'AH')
        if idx > 0:
            prev_frame = sparsegram[idx - 1]
            prev_type = prev_frame.get('phoneme_type', PhonemeType.SILENCE)
            curr_type = frame.get('phoneme_type', PhonemeType.VOWEL)
            if prev_type == PhonemeType.NASAL and curr_type == PhonemeType.VOWEL:
                if 'antiformants' not in frame:
                    frame['antiformants'] = []
                if prev_frame.get('antiformants'):
                    for af in prev_frame['antiformants']:
                        frame['antiformants'].append({
                            'fc': af['fc'],
                            'bw': af['bw'] * 1.5,
                            'decay': 0.85
                        })
        return frame


class BreathGroup:
    def __init__(self, text, punctuation):
        self.text = text.strip()
        self.punct = punctuation
        self.words = self.text.split()


class NLPParser:
    def __init__(self, base_freq=130.0, base_speed=0.25, voice_type='tenor'):
        self.f0 = base_freq
        self.speed = base_speed
        self.register = RegisterModel(voice_type)
        self.formes = create_formes_system()
        self.coarticulation = CoarticulationEngine(lookahead_frames=2, lookbehind_frames=1)

    def parse_sentences(self, raw_text):
        parts = re.split(r'([.,?!;:])', raw_text)
        groups = []
        current_text = ""
        for part in parts:
            if part in ['.', ',', '?', '!', ';', ':']:
                if current_text.strip():
                    groups.append(BreathGroup(current_text, part))
                current_text = ""
            else:
                current_text += part
        if current_text.strip():
            groups.append(BreathGroup(current_text, '.'))
        return groups

    def _text_to_phonemes(self, word: str) -> List[str]:
        phonemes = []
        word_lower = word.lower()
        i = 0
        while i < len(word_lower):
            if i < len(word_lower) - 1:
                digraph = word_lower[i:i + 2]
                if digraph in DIGRAPH_MAP:
                    phonemes.append(DIGRAPH_MAP[digraph])
                    i += 2
                    continue
            char = word_lower[i]
            if char.isalpha():
                phonemes.append(SIMPLE_DICT.get(char, 'SIL'))
            i += 1
        return phonemes

    def _is_stressed(self, word: str, word_idx: int, total_words: int) -> bool:
        if len(word) > 4:
            return True
        if word_idx == 0 or word_idx == total_words - 1:
            return True
        return False

    def generate_sparsegram(self, groups: List[BreathGroup]) -> List[Dict]:
        formes_events: List[FORMESEvent] = []
        current_time = 0.0
        formes_events.append(FORMESEvent(
            start_time=0.0,
            duration=0.2,
            event_type='phoneme',
            params={'phoneme': 'SIL', 'f0': self.f0, 'vib_depth': 0, 'effort': 0.3, 'phoneme_type': PhonemeType.SILENCE}
        ))
        current_time += 0.2
        for group in groups:
            start_pitch, end_pitch = self._get_contour(group.punct)
            pause_dur = 0.6 if group.punct in ['.', '?', '!'] else 0.3
            total_words = len(group.words)
            if total_words == 0:
                continue
            phrase_event = FORMESEvent(
                start_time=current_time,
                duration=0.0,
                event_type='phrase',
                params={'punct': group.punct}
            )
            for w_idx, word in enumerate(group.words):
                phonemes = self._text_to_phonemes(word)
                if not phonemes:
                    continue
                phrase_progress = w_idx / max(1, total_words - 1)
                word_pitch = start_pitch + (end_pitch - start_pitch) * phrase_progress
                word_effort = 0.5 + 0.3 * np.sin(np.pi * phrase_progress)
                is_stressed = self._is_stressed(word, w_idx, total_words)
                is_phrase_final = (w_idx == total_words - 1)
                word_dur_scale = 1.0 / np.log2(len(phonemes) + 2)
                for ph_idx, ph in enumerate(phonemes):
                    spec = PHONEME_TABLE.get(ph, PHONEME_TABLE['SIL'])
                    ph_pitch = word_pitch
                    ph_effort = word_effort
                    if spec.phoneme_type != PhonemeType.VOWEL:
                        ph_pitch *= 0.97
                        ph_effort *= 0.75
                    dur = self.speed * word_dur_scale
                    if spec.phoneme_type == PhonemeType.PLOSIVE:
                        dur *= 0.55
                    elif spec.phoneme_type == PhonemeType.FRICATIVE:
                        dur *= 0.85
                    vib_depth = 0.015 if spec.phoneme_type == PhonemeType.VOWEL else 0.003
                    phoneme_event = FORMESEvent(
                        start_time=current_time,
                        duration=dur,
                        event_type='phoneme',
                        params={
                            'phoneme': ph,
                            'f0': ph_pitch,
                            'vib_depth': vib_depth,
                            'effort': ph_effort,
                            'phoneme_type': spec.phoneme_type,
                            'phrase_position': phrase_progress,
                            'stressed': is_stressed and ph_idx == 0,
                            'phrase_final': is_phrase_final and ph_idx == len(phonemes) - 1,
                            'voicing': spec.voicing
                        }
                    )
                    phrase_event.children.append(phoneme_event)
                    current_time += dur
            phrase_event.duration = current_time - phrase_event.start_time
            formes_events.append(phrase_event)
            formes_events.append(FORMESEvent(
                start_time=current_time + 0.05,
                duration=pause_dur,
                event_type='phoneme',
                params={'phoneme': 'SIL', 'f0': end_pitch, 'vib_depth': 0, 'effort': 0.2, 'phoneme_type': PhonemeType.SILENCE}
            ))
            current_time += pause_dur
        scheduled = self.formes.schedule(formes_events)
        sparsegram = self._events_to_sparsegram(scheduled)
        sparsegram = self.coarticulation.apply_coarticulation(sparsegram)
        return sparsegram

    def _events_to_sparsegram(self, events: List[FORMESEvent]) -> List[Dict]:
        sparsegram: List[Dict] = []
        for event in events:
            if event.event_type != 'phoneme':
                continue
            params = event.params
            ph = params.get('phoneme', 'SIL')
            spec = PHONEME_TABLE.get(ph, PHONEME_TABLE['SIL'])
            f0 = params.get('f0', 130.0) * (1.0 + params.get('f0_offset', 0))
            effort = params.get('effort', 0.5)
            formants = []
            for idx, (fc, bw, db) in enumerate(spec.formants):
                amp = 10 ** (db / 20.0)
                formants.append({'fc': fc, 'bw': bw, 'amp': amp, 'skirt': spec.skirt_width})
            formants = self.register.shift_formants(formants, f0, effort)
            antiformants = []
            for afc, abw in spec.antiformants:
                antiformants.append({'fc': afc, 'bw': abw})
            noise_formants = []
            if spec.phoneme_type in [PhonemeType.FRICATIVE, PhonemeType.PLOSIVE]:
                for idx, (fc, bw, db) in enumerate(spec.formants):
                    amp = 10 ** (db / 20.0) * spec.aspiration
                    noise_formants.append({'fc': fc, 'bw': bw, 'amp': amp, 'decay_rate': 0.92})
            sparsegram.append({
                'time': event.start_time,
                'duration': event.duration,
                'f0': f0,
                'vib_depth': params.get('vib_depth', 0),
                'effort': effort,
                'formants': formants,
                'antiformants': antiformants,
                'noise_formants': noise_formants,
                'phoneme': ph,
                'phoneme_type': spec.phoneme_type,
                'burst_freq': spec.burst_freq,
                'burst_bw': spec.burst_bw,
                'aspiration': spec.aspiration,
                'voicing': params.get('voicing', spec.voicing),
                'transition_smooth': params.get('transition_smooth', 0.5),
                'glottal_tension': spec.glottal_tension
            })
        sparsegram.sort(key=lambda x: x['time'])
        return sparsegram

    def _get_contour(self, punct: str) -> Tuple[float, float]:
        contours = {
            '?': (0.9, 1.35),
            '!': (1.4, 1.1),
            ',': (1.0, 1.08),
            ';': (1.0, 0.95),
            ':': (1.0, 0.95),
        }
        mult = contours.get(punct, (1.05, 0.82))
        return (self.f0 * mult[0], self.f0 * mult[1])


class NoiseFormantBank:
    def __init__(self, sr: int, n_formants: int = 5):
        self.sr = sr
        self.n_formants = n_formants
        self.filters: List[Optional[np.ndarray]] = []
        self.filter_states: List[Optional[np.ndarray]] = []
        self.current_params: List[Dict] = []
        self.target_params: List[Dict] = []
        self.interp_rate = 0.002
        self._prev_filter_params: Dict[int, Tuple[float, float]] = {}
        for _ in range(n_formants):
            self.filters.append(None)
            self.filter_states.append(None)
            self.current_params.append({'fc': 1000, 'bw': 200, 'amp': 0.0})
            self.target_params.append({'fc': 1000, 'bw': 200, 'amp': 0.0})

    def set_targets(self, formants: List[Dict]):
        for i in range(min(len(formants), self.n_formants)):
            self.target_params[i] = formants[i].copy()
        for i in range(len(formants), self.n_formants):
            self.target_params[i] = {'fc': 1000, 'bw': 200, 'amp': 0.0}

    def _update_filter(self, idx: int):
        params = self.current_params[idx]
        fc = params['fc']
        bw = params['bw']
        prev_fc, prev_bw = self._prev_filter_params.get(idx, (0.0, 0.0))
        if abs(fc - prev_fc) < 5 and abs(bw - prev_bw) < 5 and self.filters[idx] is not None:
            return
        self._prev_filter_params[idx] = (fc, bw)
        if fc < 50 or fc > self.sr / 2 - 100 or bw < 10:
            self.filters[idx] = None
            return
        low = max(0.01, (fc - bw / 2) / (self.sr / 2))
        high = min(0.99, (fc + bw / 2) / (self.sr / 2))
        if low >= high:
            self.filters[idx] = None
            return
        try:
            sos = butter(2, [low, high], btype='band', output='sos')
            zi = sosfilt_zi(sos)
            self.filters[idx] = sos
            if self.filter_states[idx] is None:
                self.filter_states[idx] = zi * 0
        except Exception:
            self.filters[idx] = None

    def process_block(self, noise: np.ndarray, n_samples: int) -> np.ndarray:
        output = np.zeros(n_samples)
        for i in range(self.n_formants):
            curr = self.current_params[i]
            targ = self.target_params[i]
            curr['fc'] += (targ['fc'] - curr['fc']) * self.interp_rate * n_samples
            curr['bw'] += (targ['bw'] - curr['bw']) * self.interp_rate * n_samples
            curr['amp'] += (targ['amp'] - curr['amp']) * self.interp_rate * n_samples
            decay_rate = targ.get('decay_rate', 1.0)
            curr['amp'] *= decay_rate ** (n_samples / self.sr)
            self._update_filter(i)
            if self.filters[i] is not None and curr['amp'] > 0.001:
                try:
                    filtered, self.filter_states[i] = sosfilt(
                        self.filters[i], noise, zi=self.filter_states[i]
                    )
                    output += filtered * curr['amp']
                except Exception:
                    pass
        return output

    def reset(self):
        for i in range(self.n_formants):
            self.filter_states[i] = None
            self.current_params[i] = {'fc': 1000, 'bw': 200, 'amp': 0.0}
        self._prev_filter_params = {}


class CHANTEngine:
    def __init__(self, sr: int = 44100):
        self.sr = sr
        self.jitter = JitterGenerator(sr)
        self.noise_bank = NoiseFormantBank(sr, n_formants=5)
        self.antiformant_sos: List[np.ndarray] = []
        self.antiformant_states: List[np.ndarray] = []
        self.prev_af_hash: Optional[Tuple] = None
        self.max_grain_len = min(8000, int(0.2 * sr))
        self.grain_buffer = np.zeros(self.max_grain_len)
        self.t_table = np.arange(self.max_grain_len) / self.sr

    def _fof_grain(self, fc, bw, amp, phase, f0, skirt=1.0, tension=0.5):
        if amp < 0.0001 or fc < 20:
            return np.zeros(1)
        alpha = np.pi * bw * (0.8 + tension * 0.4)
        beta = max(fc * 0.5 * skirt, 100)
        tex = min(np.pi / beta, 0.012)
        decay_threshold = 0.001
        decay_time = min(-np.log(decay_threshold) / (alpha + 1e-9), 0.15)
        total_duration = tex + decay_time
        total_samples = int(total_duration * self.sr)
        total_samples = min(total_samples, self.max_grain_len)
        if total_samples < 2:
            return np.zeros(1)
        t = self.t_table[:total_samples]
        tex_samples = int(tex * self.sr)
        tex_samples = max(1, min(tex_samples, total_samples - 1))
        envelope = np.ones(total_samples)
        attack_phase = np.linspace(0.0, np.pi, tex_samples, endpoint=True)
        envelope[:tex_samples] = 0.5 * (1.0 - np.cos(attack_phase))
        decay = np.exp(-alpha * t)
        omega = 2.0 * np.pi * fc
        sine = np.sin(omega * t + phase)
        grain = sine * envelope * decay * amp
        if tension < 0.4:
            breath = np.random.randn(total_samples) * (0.4 - tension) * 0.1
            grain += breath * envelope * decay * amp
        return grain

    def _setup_antiformants(self, antiformants: List[Dict]):
        self.antiformant_sos = []
        self.antiformant_states = []
        for af in antiformants:
            fc = af['fc']
            bw = af['bw']
            if fc < 50 or fc > self.sr / 2 - 100:
                continue
            low = max(0.01, (fc - bw) / (self.sr / 2))
            high = min(0.99, (fc + bw) / (self.sr / 2))
            if low >= high:
                continue
            try:
                sos = butter(2, [low, high], btype='bandstop', output='sos')
                zi = sosfilt_zi(sos) * 0
                self.antiformant_sos.append(sos)
                self.antiformant_states.append(zi)
            except Exception:
                pass

    def _apply_antiformants_continuous(self, signal: np.ndarray) -> np.ndarray:
        result = signal
        for i, sos in enumerate(self.antiformant_sos):
            try:
                result, self.antiformant_states[i] = sosfilt(sos, result, zi=self.antiformant_states[i])
            except Exception:
                pass
        return result

    def _plosive_burst(self, burst_freq: float, burst_bw: float, aspiration: float, effort: float) -> np.ndarray:
        burst_duration = 0.012
        burst_samples = int(burst_duration * self.sr)
        burst = np.random.randn(burst_samples)
        burst_env = np.exp(-np.arange(burst_samples) / (burst_samples * 0.25))
        burst *= burst_env
        if burst_freq > 50 and burst_freq < self.sr / 2 - 100:
            low = max(0.01, (burst_freq - burst_bw / 2) / (self.sr / 2))
            high = min(0.99, (burst_freq + burst_bw / 2) / (self.sr / 2))
            if low < high:
                try:
                    b, a = butter(2, [low, high], btype='band')
                    burst = lfilter(b, a, burst)
                except Exception:
                    pass
        return burst * effort * 0.9

    def _interpolate_frames(self, frame_a: Dict, frame_b: Dict, frac: float, smooth: float = 0.5) -> Tuple:
        frac_smooth = 0.5 - 0.5 * np.cos(np.pi * frac)
        frac_smooth = frac_smooth ** (1.0 / (smooth + 0.5))
        f0 = frame_a['f0'] + (frame_b['f0'] - frame_a['f0']) * frac_smooth
        vib = frame_a['vib_depth'] + (frame_b['vib_depth'] - frame_a['vib_depth']) * frac_smooth
        effort = frame_a['effort'] + (frame_b['effort'] - frame_a['effort']) * frac_smooth
        voicing = frame_a.get('voicing', 1.0) + (frame_b.get('voicing', 1.0) - frame_a.get('voicing', 1.0)) * frac_smooth
        formants = []
        n_formants = min(len(frame_a['formants']), len(frame_b['formants']))
        for i in range(n_formants):
            fa = frame_a['formants'][i]
            fb = frame_b['formants'][i]
            formants.append({
                'fc': fa['fc'] + (fb['fc'] - fa['fc']) * frac_smooth,
                'bw': fa['bw'] + (fb['bw'] - fa['bw']) * frac_smooth,
                'amp': fa['amp'] + (fb['amp'] - fa['amp']) * frac_smooth,
                'skirt': fa.get('skirt', 1.0) + (fb.get('skirt', 1.0) - fa.get('skirt', 1.0)) * frac_smooth
            })
        antiformants = []
        af_a = frame_a.get('antiformants', [])
        af_b = frame_b.get('antiformants', [])
        n_anti = max(len(af_a), len(af_b))
        for i in range(n_anti):
            if i < len(af_a) and i < len(af_b):
                aa = af_a[i]
                ab = af_b[i]
                antiformants.append({
                    'fc': aa['fc'] + (ab['fc'] - aa['fc']) * frac_smooth,
                    'bw': aa['bw'] + (ab['bw'] - aa['bw']) * frac_smooth
                })
            elif i < len(af_a):
                antiformants.append({
                    'fc': af_a[i]['fc'],
                    'bw': af_a[i]['bw'] * (1 + frac_smooth * 2),
                })
            else:
                antiformants.append({
                    'fc': af_b[i]['fc'],
                    'bw': af_b[i]['bw'] * (1 + (1 - frac_smooth) * 2),
                })
        noise_formants = []
        nf_a = frame_a.get('noise_formants', [])
        nf_b = frame_b.get('noise_formants', [])
        n_noise = max(len(nf_a), len(nf_b))
        for i in range(n_noise):
            if i < len(nf_a) and i < len(nf_b):
                na = nf_a[i]
                nb = nf_b[i]
                noise_formants.append({
                    'fc': na['fc'] + (nb['fc'] - na['fc']) * frac_smooth,
                    'bw': na['bw'] + (nb['bw'] - na['bw']) * frac_smooth,
                    'amp': na['amp'] + (nb['amp'] - na['amp']) * frac_smooth,
                    'decay_rate': na.get('decay_rate', 0.95)
                })
            elif i < len(nf_a):
                decay_rate = nf_a[i].get('decay_rate', 0.92)
                noise_formants.append({
                    'fc': nf_a[i]['fc'],
                    'bw': nf_a[i]['bw'],
                    'amp': nf_a[i]['amp'] * (decay_rate ** (frac_smooth * 20)),
                    'decay_rate': decay_rate
                })
            else:
                ramp = frac_smooth ** 2
                noise_formants.append({
                    'fc': nf_b[i]['fc'],
                    'bw': nf_b[i]['bw'],
                    'amp': nf_b[i]['amp'] * ramp,
                    'decay_rate': nf_b[i].get('decay_rate', 0.95)
                })
        aspiration = frame_a.get('aspiration', 0) + (frame_b.get('aspiration', 0) - frame_a.get('aspiration', 0)) * frac_smooth
        tension = frame_a.get('glottal_tension', 0.5) + (frame_b.get('glottal_tension', 0.5) - frame_a.get('glottal_tension', 0.5)) * frac_smooth
        return f0, vib, effort, voicing, formants, antiformants, noise_formants, aspiration, tension

    def render(self, sparsegram: List[Dict]) -> np.ndarray:
        if len(sparsegram) < 2:
            return np.zeros(self.sr)
        duration = sparsegram[-1]['time'] + sparsegram[-1].get('duration', 0.5) + 0.5
        total_samples = int(duration * self.sr)
        output = np.zeros(total_samples + self.sr)
        sample_ptr = 0
        sg_idx = 0
        lfo_phase = 0.0
        lfo_freq = 5.5
        phase_acc = 0.0
        vib_rate_scale = 1.0
        vib_rate_target = 1.0
        vib_rate_counter = 0
        vib_rate_hold = int(self.sr * 0.12)
        while sample_ptr < total_samples:
            t_now = sample_ptr / self.sr
            while sg_idx < len(sparsegram) - 1 and sparsegram[sg_idx + 1]['time'] <= t_now:
                sg_idx += 1
            frame_a = sparsegram[sg_idx]
            frame_b = sparsegram[min(sg_idx + 1, len(sparsegram) - 1)]
            dt = frame_b['time'] - frame_a['time']
            frac = (t_now - frame_a['time']) / dt if dt > 0 else 0.0
            frac = float(np.clip(frac, 0.0, 1.0))
            smooth = frame_a.get('transition_smooth', 0.6)
            smooth = max(0.6, smooth)
            f0_base, vib_depth, effort, voicing, formants, antiformants, noise_formants, aspiration, tension = \
                self._interpolate_frames(frame_a, frame_b, frac, smooth)
            jitter_val = self.jitter.next()
            vib_val = np.sin(lfo_phase) * vib_depth
            current_f0 = f0_base * (1.0 + jitter_val + vib_val)
            current_f0 = max(20.0, min(current_f0, 2000.0))
            period_len = max(10, int(self.sr / current_f0))
            phase_acc += current_f0 / self.sr
            af_hash = tuple((round(af['fc'], 3), round(af['bw'], 3)) for af in antiformants) if antiformants else ()
            vib_rate_counter -= 1
            if vib_rate_counter <= 0:
                vib_rate_target = 1.0 + np.random.uniform(-0.04, 0.04)
                vib_rate_counter = vib_rate_hold
            vib_rate_scale += 0.0015 * (vib_rate_target - vib_rate_scale)
            if phase_acc >= 1.0:
                phase_acc -= 1.0
                if af_hash != self.prev_af_hash:
                    self._setup_antiformants(antiformants)
                    self.prev_af_hash = af_hash
                ph_type = frame_a.get('phoneme_type', PhonemeType.VOWEL)
                grain_output = self.grain_buffer
                grain_output.fill(0.0)
                used_len = 0
                if ph_type == PhonemeType.PLOSIVE and frac < 0.3:
                    burst_freq = frame_a.get('burst_freq', 1000)
                    burst_bw = frame_a.get('burst_bw', 500)
                    burst = self._plosive_burst(burst_freq, burst_bw, aspiration, effort)
                    burst_len = len(burst)
                    grain_output[:burst_len] += burst
                    used_len = max(used_len, burst_len)
                if voicing > 0.05 and ph_type != PhonemeType.SILENCE:
                    for i, fmt in enumerate(formants):
                        phase = i * np.pi * 0.4
                        shimmer = np.random.uniform(0.96, 1.04)
                        grain = self._fof_grain(
                            fmt['fc'],
                            fmt['bw'],
                            fmt['amp'] * shimmer * effort * voicing,
                            phase,
                            current_f0,
                            fmt.get('skirt', 1.0),
                            tension
                        )
                        g_len = len(grain)
                        grain_output[:g_len] += grain
                        used_len = max(used_len, g_len)
                if antiformants and voicing > 0.05:
                    active_len = min(self.max_grain_len, max(period_len * 2, used_len))
                    grain_output[:active_len] = self._apply_antiformants_continuous(grain_output[:active_len])
                    used_len = max(used_len, active_len)
                if noise_formants and (aspiration > 0.01 or ph_type in [PhonemeType.FRICATIVE, PhonemeType.PLOSIVE]):
                    self.noise_bank.set_targets(noise_formants)
                    noise_block = max(64, min(2048, period_len))
                    noise_input = np.random.randn(noise_block)
                    noise_output = self.noise_bank.process_block(noise_input, noise_block)
                    noise_output *= effort * (1.0 - voicing * 0.5)
                    n_len = len(noise_output)
                    grain_output[:n_len] += noise_output
                    used_len = max(used_len, n_len)
                if used_len > 0 and period_len > 0:
                    overlaps = max(1, int(np.ceil(used_len / period_len)))
                    grain_output /= np.sqrt(overlaps)
                if used_len == 0:
                    continue
                grain_output = np.nan_to_num(grain_output, nan=0.0, posinf=0.0, neginf=0.0)
                end_idx = min(sample_ptr + used_len, len(output))
                if end_idx <= sample_ptr:
                    continue
                segment = output[sample_ptr:end_idx] + grain_output[:end_idx - sample_ptr]
                clip_thresh = 8.0
                if np.max(np.abs(segment)) > clip_thresh:
                    segment = np.tanh(segment / clip_thresh) * clip_thresh
                output[sample_ptr:end_idx] = segment
            lfo_phase += 2.0 * np.pi * lfo_freq * vib_rate_scale / self.sr
            if lfo_phase > 2.0 * np.pi:
                lfo_phase -= 2.0 * np.pi
            sample_ptr += 1
        output = output[:total_samples]
        peak = np.max(np.abs(output))
        if peak > 0:
            output = output / peak * 0.9
        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CHANT-style FOF synthesis with coarticulation and FORMES")
    parser.add_argument('text', type=str, help='input text with punctuation')
    parser.add_argument('-o', '--out', default='', help='output file (default: derived from text)')
    parser.add_argument('-f', '--pitch', type=float, default=0.0, help='base pitch Hz (0=auto from voice)')
    parser.add_argument('-s', '--speed', type=float, default=0.38, help='phoneme duration scale')
    parser.add_argument('-v', '--voice', default='tenor', choices=['soprano', 'alto', 'tenor', 'bass'], help='voice type')
    parser.add_argument('-t', '--tempo', type=float, default=1.0, help='tempo multiplier')
    parser.add_argument('-d', '--dynamics', type=float, default=0.7, help='dynamics 0-1')
    parser.add_argument('--lookahead', type=int, default=8, help='coarticulation lookahead frames')
    parser.add_argument('--phonemes', default='', help='path to phoneme YAML/JSON table')
    parser.add_argument('--text-file', default='', help='path to text file to synthesize')
    args = parser.parse_args()

    input_text = args.text
    if args.text_file:
        try:
            input_text = Path(args.text_file).read_text(encoding='utf-8')
        except Exception:
            input_text = args.text

    auto_pitch = {'soprano': 440.0, 'alto': 330.0, 'tenor': 220.0, 'bass': 165.0}
    base_pitch = args.pitch if args.pitch > 0 else auto_pitch.get(args.voice, 220.0)

    out_path = args.out
    if not out_path:
        clean = re.sub(r'[^A-Za-z0-9]+', '_', input_text.strip())
        clean = clean.strip('_')
        if len(clean) == 0:
            clean = "chant"
        out_path = f"{clean[:24]}.wav"

    if args.phonemes:
        try:
            globals()['PHONEME_TABLE'] = load_phoneme_table(Path(args.phonemes))
        except Exception:
            # fall back silently to default
            globals()['PHONEME_TABLE'] = load_phoneme_table()

    nlp = NLPParser(base_freq=base_pitch, base_speed=args.speed, voice_type=args.voice)
    nlp.formes.context['tempo'] = args.tempo
    nlp.formes.context['dynamics'] = args.dynamics
    nlp.coarticulation.lookahead = args.lookahead

    breath_groups = nlp.parse_sentences(input_text)
    sparsegram = nlp.generate_sparsegram(breath_groups)

    engine = CHANTEngine(sr=44100)
    audio = engine.render(sparsegram)
    wav.write(out_path, 44100, (audio * 32767).astype(np.int16))
