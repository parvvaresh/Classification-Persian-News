import { Clapperboard, Coins, Landmark, Tag, Trophy, Users } from "lucide-react";

const LABEL_ICONS = {
  sports: Trophy,
  social: Users,
  "culture-media": Clapperboard,
  economy: Coins,
  politics: Landmark,
};

export const labelIcon = (label) => LABEL_ICONS[label] ?? Tag;
