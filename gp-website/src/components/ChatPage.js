import React from 'react'
import '../styles/Chat.css'
import NavBar from './navbar.js'
import robot from '../assets/loveBot.svg'

import { WindupChildren } from "windups";

import {
  MainContainer,
  ChatContainer,
  MessageList,
  Message
} from "@chatscope/chat-ui-kit-react";

const ChatPage = () => {
  const [messages, setMessages] = React.useState([
    {
      message: "Hello, I'm Maven, your personal health assistant. How can I help you today?",
      sender: 'Maven'
    }
  ])

  const handleYes = async () => {
    const newMessage = {
      message: "Yes",
      sender: 'user',
      direction: 'outgoing'
    }

    const newMessages = [...messages, newMessage]

    setMessages(newMessages)

    const response = {
      message: "Great! Let's get started. What symptoms are you experiencing?",
      sender: 'Maven'
    }

    const updatedMessages = [...newMessages, response]
    setMessages(updatedMessages)
  

  }
  const handleNo = () => {}
 

  return (
    <div>
        <NavBar sticky={1}/>
        <div className='chat-section'>
          <div className='warning-message chat-content'>
            <p> <span>Warning: </span> This tool is not a substitute for professional medical advice, diagnosis, or treatment. If you are experiencing a life-threatening emergency that requires immediate attention please call 123 or the number for your local emergency service.</p>
          </div>
          <div className='interactice-section'> 
          </div>

          <div className='interactice-section'>
            <MainContainer>
              <ChatContainer>
                <MessageList>
                    {
                      messages.map((message, index) => {
                        if (message.sender === 'Maven') {
                            return (
                              <div>
                                <div className='message-wrapper'>
                                  <img src={robot} alt='robot' className='robot'/>
                                  <Message className = "message" key={index} model={message}/>
                                </div>
                                <div className='answer-wrapper'>
                                  <button className='btn yes' onClick={handleYes}>Yes</button>
                                  <button className='btn no' onClick={handleNo}>No</button>
                                </div>
                              </div>
                            );
                        }
                        else {
                          return (
                            <Message className = "answer-message" key={index} model={message}/>
                          );
                        }
                      })
                    }
                </MessageList>
              </ChatContainer>
            </MainContainer>
          </div>


        </div>
  
    </div>
  )
}

export default ChatPage