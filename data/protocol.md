# Sky Protocol: Ground Truth Classification Rules

### Classes (4 total):
1. **0: Clear (`clear`)**
   - Unobstructed blue or night sky.
   - Less than ~10% cloud coverage. Sun or moon disk is sharply defined with crisp shadows.
2. **1: Scattered Clouds (`scattered`)**
   - Cumulus, stratocumulus, or cirrus clouds covering between 10% and 80% of the visible frame.
   - Distinct patches of blue sky remain visible.
3. **2: Overcast (`overcast`)**
   - Uniform cloud layer covering >80% of the frame.
   - Gray or white sheet. No blue sky visible.
4. **3: Hazy (`hazy`)**
   - Reduced visibility due to particulate matter, dust, fog, or heavy smog.
   - Pale, milky appearance where blue sky is washed out, but distinct cloud boundaries are absent.

### Camera Protocol for Tomorrow:
- **Angle**: Fixed 45° elevation angle pointing away from direct solar glare.
- **Stations (5 spots)**:
  1. Station 1: Window / Balcony
  2. Station 2: Narrow street / Alley
  3. Station 3: Open park / Field
  4. Station 4: Rooftop / Elevated point
  5. Station 5: Near water / Open horizon
- **Photos per station**: ~30 photos for training + 8 fresh photos for testing.
