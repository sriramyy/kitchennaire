// Fallback for using MaterialIcons on Android and web.

import MaterialIcons from '@expo/vector-icons/MaterialIcons';
import { SymbolWeight, SymbolViewProps } from 'expo-symbols';
import { ComponentProps } from 'react';
import { OpaqueColorValue, type StyleProp, type TextStyle } from 'react-native';

type IconMapping = Partial<Record<SymbolViewProps['name'], ComponentProps<typeof MaterialIcons>['name']>>;
type IconSymbolName = keyof typeof MAPPING;

/**
 * Add your SF Symbols to Material Icons mappings here.
 * - see Material Icons in the [Icons Directory](https://icons.expo.fyi).
 * - see SF Symbols in the [SF Symbols](https://developer.apple.com/sf-symbols/) app.
 */
const MAPPING = {
  // common app icons
  'link': 'link',
  'timer': 'timer',
  'video-camera': 'videocam',
  'bookmark': 'bookmark',
  'play': 'play-arrow',

  // search / camera / photo
  'magnifyingglass': 'search',
  'camera.fill': 'photo_camera',
  'photo.fill': 'photo',

  // check / x / circle
  'checkmark.circle.fill': 'check-circle',
  'checkmark.circle': 'check-circle',
  'checkmark-circle': 'check-circle',
  'x-circle': 'cancel',
  'circle': 'panorama-fish-eye',

  // other mappings used in the app
  'arrow.counterclockwise': 'refresh',
  'square.and.arrow.up': 'file_upload',
  'questionmark.circle': 'help-outline',
  'exclamationmark.circle': 'error',
  'play.slash': 'block',

  // developer / code icons
  'house.fill': 'home',
  'paperplane.fill': 'send',
  'chevron.left.forwardslash.chevron.right': 'code',
  'chevron.right': 'chevron-right',
} as unknown as IconMapping;

/**
 * An icon component that uses native SF Symbols on iOS, and Material Icons on Android and web.
 * This ensures a consistent look across platforms, and optimal resource usage.
 * Icon `name`s are based on SF Symbols and require manual mapping to Material Icons.
 */
export function IconSymbol({
  name,
  size = 24,
  color,
  style,
}: {
  name: IconSymbolName;
  size?: number;
  color: string | OpaqueColorValue;
  style?: StyleProp<TextStyle>;
  weight?: SymbolWeight;
}) {
  // Use mapped Material Icon name when available, otherwise try the incoming name as a fallback
  const mapped = MAPPING[name] as ComponentProps<typeof MaterialIcons>['name'] | undefined;
  const iconName = mapped ?? (name as unknown as ComponentProps<typeof MaterialIcons>['name']);
  return <MaterialIcons color={color} size={size} name={iconName} style={style} />;
}
