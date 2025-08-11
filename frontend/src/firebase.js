// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
import { getAuth } from "firebase/auth";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyCbVaEOXX40bUNlJm7QMwug1U82vZdTxkM",
  authDomain: "finbuddy-9edbd.firebaseapp.com",
  projectId: "finbuddy-9edbd",
  storageBucket: "finbuddy-9edbd.firebasestorage.app",
  messagingSenderId: "150285490550",
  appId: "1:150285490550:web:ecc662001a2e7025e98257",
  measurementId: "G-76PXXVHDRD"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);
export const auth = getAuth(app); 