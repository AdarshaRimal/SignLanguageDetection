/* eslint-disable react/no-unescaped-entities */
/* eslint-disable no-unused-vars */
import React, { useState } from 'react';
import './Dashboard.css';

const Dashboard = () => {
  // const user = {
  //   name: 'John Doe',
  //   language: 'English',
  //   profilePicture: 'https://cdn.pixabay.com/photo/2024/11/22/13/20/man-9216455_1280.jpg', // Placeholder for profile picture
  // };

  const gestureLibrary = [
    { id: 1, name: 'Hello', icon: '🖐️' },
    { id: 2, name: 'Thank You', icon: '🙏' },
    { id: 3, name: 'Goodbye', icon: '👋' },
    { id: 4, name: 'Yes', icon: '👍' },
    { id: 5, name: 'No', icon: '👎' },
  ];

  const [selectedGesture, setSelectedGesture] = useState(null);

  return (
    <div className="dashboard">
      {/* User Profile Card */}
      {/* <div className="profile-card">
        <img src={user.profilePicture} alt="Profile" className="profile-picture" />
        <h2>{user.name}</h2>
        <p><strong>Language:</strong> {user.language}</p>
      </div> */}

      {/* Gesture Library Section */}
      <div className="gesture-library">
        <center><h3>Gesture Library</h3></center>
        <div className="gesture-grid">
          {gestureLibrary.map((gesture) => (
            <div
              key={gesture.id}
              className="gesture-card"
              onClick={() => setSelectedGesture(gesture)}
            >
              <span className="gesture-icon">{gesture.icon}</span>
              <p>{gesture.name}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Practice Section */}
      <div className="practice-section">
        <h3>Practice Gesture</h3>
        {selectedGesture ? (
          <div className="practice-feedback">
            <h4>Practicing: {selectedGesture.name}</h4>
            <p>Perform the gesture "{selectedGesture.name}" using your camera.</p>
          </div>
        ) : (
          <p>Select a gesture from the library to start practicing.</p>
        )}
      </div>
    </div>
  );
};

export default Dashboard;
