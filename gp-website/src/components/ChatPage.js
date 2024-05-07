import React, { useEffect } from 'react'
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
import { redirect } from 'react-router-dom';

const ChatPage = () => {
  // const [idx, setIdx] = React.useState(0)
  const [messages, setMessages] = React.useState([
    // {
    //   message: "Hello, I'm Maven, your personal health assistant. How can I help you today?",
    //   sender: 'Maven'
    // }
  ])
  const [diagnosed, setDiagnosed] = React.useState(false)
  const currentLink = window.location.href

  // const typeWritter = (index, msg, msgIndex) => {
  //   setTimeout(() => {
  //     //get index of the message 
  //     var newMessages = messages;
  //     newMessages[index] = {message: msg.substr(0,msgIndex), sender: 'Maven'} 
  //     setMessages([...newMessages])
  //     if ( msgIndex < msg.length) {
  //       typeWritter(index, msg, msgIndex+1)
  //     }
  //   }, 23)
  // }
   
  const fetchFirstMessage = async () => {
    let symptom = currentLink.split('/').pop()
    const response = await fetch('http://127.0.0.1:8000/maven/'+symptom)
    const data = await response.json()
    // console.log(data)
    const message = {
      message: data.Question,
      sender: 'Maven'
    }

    const newMessages = [...messages, message]
    setMessages(newMessages)

    // typeWritter(idx, data.Question, 0)
  }
  useEffect(() => {
    fetchFirstMessage()
  }, [])




  async function fetchNextQuestion (answer, updateMsgs) {
    let symptom = currentLink.split('/').pop();
    let url = 'http://127.0.0.1:8000/maven/'+symptom;

    fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json', //specify the content type
      },
      body: JSON.stringify({ans: answer}),
    }).then(response => response.json())
    .then(data => {
      const Q = data.Question
      // check if the word Diagnosis: is in the question
      if (Q.includes('Diagnosis:')) {
        setDiagnosed(true)
      }
      const m = {
        message: Q,
        sender: 'Maven'
      }
      const newM = [...updateMsgs, m];
      setMessages(newM)
      // setIdx(idx+1)
      // typeWritter(idx, Q,0)
    })
    //get data from promise 
  }

  const handleYes = async () => {
    const newMessage = {
      message: "Yes",
      sender: 'user',
      direction: "outgoing"
    }
    
    //before adding the answer I want to set the previous question to be answered
    const lastQ = messages[messages.length - 1]
    const updatedLastQ = {
      ...lastQ,
      status: 'answered'
    }

    const newMessages = messages.slice(0, messages.length - 1)
    const update = [...newMessages, updatedLastQ, newMessage]
    
    setMessages(update);

    await fetchNextQuestion('yes', update); //need to pass the updated messages to the function
    // because the function will update the messages state with the new message 
    

  }
  const handleNo = async () => {
    const newMessage = {
      message: "No",
      sender: 'user',
      direction: "outgoing"
    }
    
    const lastQ = messages[messages.length - 1]
    const updatedLastQ = {
      ...lastQ,
      status: 'answered'
    }
    //remove the last question from messages
    const newMessages = messages.slice(0, messages.length - 1)
    const update = [...newMessages, updatedLastQ, newMessage]
    
    setMessages(update);

    await fetchNextQuestion('no', update); //need to pass the updated messages to the function
    // because the function will update the messages state with the new message 
    
  }
 

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
                        if (message.status === 'answered') {
                          return (
                                <div className='message-wrapper'>
                                  <img src={robot} alt='robot' className='robot'/>
                                  <Message className = "message" key={index} model={message}/>
                                </div>
                          );
                        }
                        else if (message.sender === 'Maven') {
                            return (
                              <div>
                                <div className='message-wrapper'>
                                  <img src={robot} alt='robot' className='robot'/>
                                  <Message className = "message" key={index} model={message}/>
                                </div>
                                <div className={diagnosed? 'Hidden' : 'answer-wrapper'}>
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