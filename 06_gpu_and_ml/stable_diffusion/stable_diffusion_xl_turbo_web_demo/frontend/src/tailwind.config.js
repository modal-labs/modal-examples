/* ##RefactorExamplesComponentsAndStyles */
import defaultTheme from 'tailwindcss/defaultTheme';

/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{html,js,svelte,ts}'],
  theme: {
    extend: {
      screens: {
        md: "840px",
        xl: "1200px",
      },
      colors: {
        // Dark base background color
        ground: "#0C0F0B",

        // Theme colors
        primary: "#7FEE64",
        "accent-pink": "#FF0ECA",
        "accent-blue": "#B8E4FF",
      },
      fontFamily: {
        mono: ["Fira Mono", ...defaultTheme.fontFamily.mono],
        sans: ["Inter Variable", ...defaultTheme.fontFamily.sans],
        inter: ["Inter Variable", ...defaultTheme.fontFamily.sans],
        tosh: ["Tosh Modal", ...defaultTheme.fontFamily.sans],
      },

      // Global font modifications
      fontSize: {
        sm: [
          "0.875rem",
          {
            lineHeight: "1.25rem",
            letterSpacing: "0.01em",
          },
        ],
      },
    },
  },
  plugins: [],
}

