using System;
using System.Xml.Schema;
using AI;
using Fiourp;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Input;

namespace CartPole;

public class Environment : Entity
{
    public Cart Cart;
    public Pole Pole;
    private DeepQAgent agent;
    
    private int timeStep = 0;
    private int targetTimeStep = 0;
    private int targetRefreshRate = 50;

    private int maxDistanceCar = 200;
    private bool learning = true;

    public Environment(DeepQAgent agent, bool rendered) : base(Vector2.Zero)
    {
        this.agent = agent;
        Cart = new Cart();
        Pole = new Pole(Cart);
        
        Visible = rendered;
    }

    public override void Update()
    {
        base.Update();

        float[] state = GetState();
        int action = agent.Act(state);
        Debug.LogUpdate(action);
        Cart.Update(action);
        Pole.Update();
        float[] nextState = GetState();
        
        bool done = Cart.MiddlePos.X > Engine.ScreenSize.X / 2 + maxDistanceCar || Cart.MiddlePos.X < Engine.ScreenSize.X / 2 - maxDistanceCar || Pole.Pos.Y > Cart.Pos.Y - 200 + 60 || timeStep >= 1000;

        float reward = 1;
        if (done)
            reward = -30;

        /*for(int i = 0; i < state.Length; i++)
            Debug.LogUpdate(state[i]);*/

        agent.Remember(state, action, reward, nextState, done);

        agent.Replay();

        if (done)
        {
            if (timeStep > 100)
                Console.ForegroundColor = ConsoleColor.Green;
            else if (timeStep > 70)
                Console.ForegroundColor = ConsoleColor.Yellow;
            else
                Console.ForegroundColor = ConsoleColor.Red;
            
            Console.WriteLine($"Episode {Main.episode}, Score {timeStep}, Epsilon: {agent.Epsilon}");
            Console.ForegroundColor = ConsoleColor.Gray;

            Main.episode++;
            timeStep = 0;

            Cart.Reset();
            Pole.Reset();
        }

        if (Input.GetKeyDown(Keys.S) && Visible)
            agent.Network.Save("/home/f/Documents/CartPoleDeepQ/saves/net");
        if (Input.GetKeyDown(Keys.L) && Visible)
            agent.Network.Load("/home/f/Documents/CartPoleDeepQ/saves/net");

        /*
        if (Game1.episode > agent.totalEpisodes)
        {
            agent.Network.Save("/home/f/Documents/CartPoleDeepQ/saves/netSaved");
            Game1.game1.Exit();
        }
        */
        
        timeStep++;

        if (Visible)
        {
            targetTimeStep++;
            if (targetTimeStep >= agent.targetRefreshRate)
            {
                agent.RefreshTargetNetwork();
                targetTimeStep = 0;
            }
        }
    }

    public override void Render()
    {
        base.Render();
        
        Cart.Render();
        Pole.Render();
    }

    public float[] GetState()
    {
        //Debug.LogUpdate("Cart Pos: " + (Cart.MiddlePos.X - Engine.ScreenSize.X / 2), "Cart Velocity: " + Cart.Velocity, "Pole Angle: " + Pole.GetAngle(out _), "Pole gap: " + (Pole.Pos.X - Cart.MiddlePos.X));
        
        /*Console.WriteLine("Cart Pos: " + Normalize(Cart.MiddlePos.X - Engine.ScreenSize.X / 2, -maxDistanceCar, maxDistanceCar) +
                          "\nCart Velocity: " + Normalize(Cart.Velocity, -20, 20) +
                          "\nPole Angle: " + Normalize(Pole.GetAngle(out _), -0.72f, 0.72f) +
                          "\nPole gap: " + Normalize(Pole.Pos.X - Cart.MiddlePos.X, -130, 130));*/
        
        return new float[4]
        {
            Normalize(Cart.MiddlePos.X - Engine.ScreenSize.X / 2, -maxDistanceCar, maxDistanceCar),
            Normalize(Cart.Velocity, -20, 20),
            Normalize(Pole.GetAngle(out _), -0.72f, 0.72f),
            Normalize(Pole.Pos.X - Cart.MiddlePos.X, -130, 130)
        };
    }

    public float Normalize(float value, float min, float max)
        => (value - min) / (max - min);
}