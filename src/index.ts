import build from './builder';

try {
  build();
} catch (err) {
  console.error(err);
}
